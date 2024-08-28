"""
YOLO training.

This code allows you to train an object detection model with the YOLOv8 neck and loss.

To run this script, you can start it with:
    python train.py cfg/yolo_phinet.py

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023
"""

import torch
import torch.nn as nn
import torch.optim as optim
from prepare_data import create_loaders
from yolo_loss import Loss
import math

import micromind as mm
from micromind.networks import PhiNet
from micromind.networks.yolo import SPPF, Yolov8Neck, DetectionHead
from micromind.utils import parse_configuration
from micromind.utils.yolo import load_config
import sys
import os
from micromind.utils.yolo import get_variant_multiples
from validation.validator import DetectionValidator

from torch.autograd import Variable
from canny import Net
from clustering import KMeans
from ultralytics import YOLO as YOLOseg

def grayscale(batch):
    with torch.no_grad():
        gray_images = 0.299 * batch[:, 0, :, :] + 0.587 * batch[:, 1, :, :] + 0.114 * batch[:, 2, :, :]
        gray_images = gray_images.unsqueeze(1)
    return gray_images

def rgb_to_hsv(batch):
    with torch.no_grad():
        r, g, b = batch[:, 0, :, :], batch[:, 1, :, :], batch[:, 2, :, :]
        max_rgb, _ = torch.max(batch, dim=1)
        min_rgb, _ = torch.min(batch, dim=1)
        delta = max_rgb - min_rgb
        hue = torch.zeros_like(max_rgb)
        mask = delta != 0
        hue[mask & (max_rgb == r)] = (60 * (g[mask & (max_rgb == r)] - b[mask & (max_rgb == r)]) / delta[mask & (max_rgb == r)]) % 360
        hue[mask & (max_rgb == g)] = (60 * (b[mask & (max_rgb == g)] - r[mask & (max_rgb == g)]) / delta[mask & (max_rgb == g)]) + 120
        hue[mask & (max_rgb == b)] = (60 * (r[mask & (max_rgb == b)] - g[mask & (max_rgb == b)]) / delta[mask & (max_rgb == b)]) + 240
        hue = hue / 360.0
        saturation = torch.zeros_like(max_rgb)
        saturation[max_rgb != 0] = delta[max_rgb != 0] / max_rgb[max_rgb != 0]
        value = max_rgb
        hsv_images = torch.stack([hue, saturation, value], dim=1)
    return hsv_images

def hsv_to_rgb(batch): # just to test
    with torch.no_grad():
        h, s, v = batch[:, 0, :, :], batch[:, 1, :, :], batch[:, 2, :, :]
        h = h * 360.0
        c = s * v
        x = c * (1 - torch.abs((h / 60.0) % 2 - 1))
        m = v - c
        z = torch.zeros_like(h)
        mask1 = (h >= 0)   & (h < 60)
        mask2 = (h >= 60)  & (h < 120)
        mask3 = (h >= 120) & (h < 180)
        mask4 = (h >= 180) & (h < 240)
        mask5 = (h >= 240) & (h < 300)
        mask6 = (h >= 300) & (h < 360)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        r[mask1] = c[mask1]
        g[mask1] = x[mask1]
        b[mask1] = z[mask1]
        
        r[mask2] = x[mask2]
        g[mask2] = c[mask2]
        b[mask2] = z[mask2]
        
        r[mask3] = z[mask3]
        g[mask3] = c[mask3]
        b[mask3] = x[mask3]
        
        r[mask4] = z[mask4]
        g[mask4] = x[mask4]
        b[mask4] = c[mask4]
        
        r[mask5] = x[mask5]
        g[mask5] = z[mask5]
        b[mask5] = c[mask5]
        
        r[mask6] = c[mask6]
        g[mask6] = z[mask6]
        b[mask6] = x[mask6]
        
        r = (r + m).clamp(0, 1)
        g = (g + m).clamp(0, 1)
        b = (b + m).clamp(0, 1)
        rgb_images = torch.stack([r, g, b], dim=1)
    return rgb_images

def canny(batch, device, use_cuda=True):
    canny_images = []
    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()
    
    with torch.no_grad():
        for i in range(len(batch)):
            data = Variable(batch[i].unsqueeze(0)).to(device)
            canny_tensor = net(data)[4]  # the thresholded one
            canny_tensor /= canny_tensor.max()
            canny_images.append(canny_tensor)
    
    return torch.cat(canny_images, dim=0)

def segment(batch, segmentation_model, device):
    results = []
    with torch.no_grad():
        for i in range(len(batch)):
            result = segmentation_model(batch[i].unsqueeze(0), verbose=False)
            try:
                result = torch.any(result[0].masks.data, dim=0, keepdim=True)
            except AttributeError:
                result = torch.zeros((1, *batch[i].shape[1:]), dtype=torch.bool)
            results.append(result.to(device))

    segmented_batch = torch.cat(results, dim=0).unsqueeze(1).float()
    return segmented_batch

def kmeans_clustering(batch, k=3, n_iters=10):
    results = []
    with torch.no_grad():
        for i in range(len(batch)):
            shape = batch[i].shape
            pixels = batch[i].permute(1, 2, 0).reshape(-1, 3)
            kmeans = KMeans(X=pixels, k=k, n_iters=n_iters, p=2)
            kmeans.train(pixels)
            labels = kmeans.predict(pixels)
            segmented_image = kmeans.train_pts[labels]
            segmented_image = segmented_image.reshape(shape[1], shape[2], shape[0]).permute(2, 0, 1)
            if torch.isnan(segmented_image).all():
                segmented_image = torch.zeros_like(segmented_image)
            results.append(segmented_image)

    clustered_batch = torch.stack(results, dim=0)
    return clustered_batch


class YOLO(mm.MicroMind):
    def __init__(self, m_cfg, hparams, *args, **kwargs):
        """Initializes the YOLO model."""
        super().__init__(*args, **kwargs)

        self.m_cfg = m_cfg
        w, r, d = get_variant_multiples("n")

        self.modules["backbone"] = PhiNet(
            input_shape=hparams.input_shape,
            alpha=hparams.alpha,
            beta=hparams.beta,
            t_zero=hparams.t_zero,
            num_layers=hparams.num_layers,
            h_swish=False,
            squeeze_excite=True,
            include_top=False,
            num_classes=hparams.num_classes,
            divisor=hparams.divisor,
            compatibility=False,
            downsampling_layers=hparams.downsampling_layers,
            return_layers=hparams.return_layers,
        )

        sppf_ch, neck_filters, up, head_filters = self.get_parameters(
            heads=hparams.heads
        )

        self.modules["sppf"] = SPPF(*sppf_ch)
        self.modules["neck"] = Yolov8Neck(
            filters=neck_filters, up=up, heads=hparams.heads
        )

        self.modules["head"] = DetectionHead(
            hparams.num_classes, filters=head_filters, heads=hparams.heads
        )
        self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)
        self.segmentation_model = YOLOseg("yolov8n-seg.pt")  # load an official model

        print("Number of parameters for each module:")
        print(self.compute_params())

    def get_parameters(self, heads=[True, True, True]):
        """
        Gets the parameters with which to initialize the network detection part
        (SPPF block, Yolov8Neck, DetectionHead).

        Arguments
        ---------
        heads : Optional[List]
            List indicating whether each detection head is active.
            Default: [True, True, True].

        Returns
        -------
        Tuple containing the parameters for initializing the network detection part.
        Contains
            - Tuple (c1, c2): Tuple of input channel sizes for the SPPF block.
            - List neck_filters: List of filter sizes for Yolov8Neck.
            - List up: List of upsampling factors for Yolov8Neck.
            - List head_filters: List of filter sizes for DetectionHead. : Tuple
        """
        in_shape = self.modules["backbone"].input_shape
        x = torch.randn(1, *in_shape)
        y = self.modules["backbone"](x)

        c1 = c2 = y[0].shape[1]
        sppf = SPPF(c1, c2)
        out_sppf = sppf(y[0])

        neck_filters = [y[1][0].shape[1], y[1][1].shape[1], out_sppf.shape[1]]
        up = [2, 2]
        up[0] = y[1][1].shape[2] / out_sppf.shape[2]
        up[1] = y[1][0].shape[2] / (up[0] * out_sppf.shape[2])
        temp = """The layers you selected are not valid. \
            Please choose only layers between which the spatial resolution \
            doubles every time. Eventually, you can achieve this by \
            changing the downsampling layers. If you are trying to change \
            the input resolution, make sure you also change it in the \
            dataset configuration file and that it is a multiple of 4."""

        assert up == [2, 2], " ".join(temp.split())

        neck = Yolov8Neck(filters=neck_filters, up=up)
        out_neck = neck(y[1][0], y[1][1], out_sppf)

        head_filters = (
            out_neck[0].shape[1],
            out_neck[1].shape[1],
            out_neck[2].shape[1],
        )
        # keep only the heads we want
        head_filters = [head for heads, head in zip(heads, head_filters) if heads]

        return (c1, c2), neck_filters, up, head_filters

    def preprocess_batch(self, batch, skip_augs=False):
        """Preprocesses a batch of images by scaling and converting to float."""
        preprocessed_batch = {}
        preprocessed_batch["img"] = (
            batch["img"].to(self.device, non_blocking=True).float() / 255
        )
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and k != "img":
                preprocessed_batch[k] = batch[k].to(self.device)

        if not skip_augs:

            hsv_batch = rgb_to_hsv(preprocessed_batch["img"])
            grayscale_batch = grayscale(preprocessed_batch["img"])
            canny_batch = canny(preprocessed_batch["img"], self.device).detach()
            segmented_batch = segment(preprocessed_batch["img"], self.segmentation_model, self.device).detach()
            clustered_batch = kmeans_clustering(preprocessed_batch["img"]).detach()

            # NOTE: this is to modify based on the type of experiment you decide to train. 
            # NOTE: This needs to be coherent with the `input_shape` parameter inside the `cfg/yolo_phinet.py` file.
            final_batch = torch.cat((preprocessed_batch["img"], hsv_batch, grayscale_batch, canny_batch, segmented_batch, clustered_batch), dim=1)
            
            breakpoint()
            preprocessed_batch["img"] = final_batch

        return preprocessed_batch

    def forward(self, batch):
        """Runs the forward method by calling every module."""
        preprocessed_batch = self.preprocess_batch(batch)
        backbone = self.modules["backbone"](
            preprocessed_batch["img"].to(self.device)
        )

        neck_input = backbone[1]
        neck_input.append(self.modules["sppf"](backbone[0]))
        neck = self.modules["neck"](*neck_input)
        head = self.modules["head"](neck)

        return head

    def compute_loss(self, pred, batch):
        """Computes the loss."""
        preprocessed_batch = self.preprocess_batch(batch, skip_augs=True)

        lossi_sum, lossi = self.criterion(
            pred,
            preprocessed_batch,
        )

        return lossi_sum

    def build_optimizer(
        self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e6
    ):
        """
        Constructs an optimizer for the given model, based on the specified optimizer
        name, learning rate, momentum, weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the
                optimizer is selected based on the number of iterations.
                Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer.
                Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines
                the optimizer if name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(
            v for k, v in nn.__dict__.items() if "Norm" in k
        )  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            print(
                f"optimizer: 'optimizer=auto' found, "
                f"ignoring 'lr0={lr}' and 'momentum={momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 80)  # number of classes
            lr_fit = round(
                0.002 * 5 / (4 + nc), 6
            )  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("AdamW", lr_fit, 0.9)
            lr *= 10
            # self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optimizer = getattr(optim, name, optim.Adam)(
                g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit"
                "https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group(
            {"params": g[0], "weight_decay": decay}
        )  # add g0 with weight_decay
        optimizer.add_param_group(
            {"params": g[1], "weight_decay": 0.0}
        )  # add g1 (BatchNorm2d weights)
        print(
            f"{optimizer:} {type(optimizer).__name__}(lr={lr}, "
            f"momentum={momentum}) with parameter groups"
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} "
            f"weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer, lr

    def _setup_scheduler(self, opt, lrf=0.01, lr0=0.01, cos_lr=True):
        """Initialize training learning rate scheduler."""

        def one_cycle(y1=0.0, y2=1.0, steps=100):
            """Returns a lambda function for sinusoidal ramp from y1 to y2
            https://arxiv.org/pdf/1812.01187.pdf."""
            return (
                lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1)
                + y1
            )

        lrf *= lr0

        if cos_lr:
            self.lf = one_cycle(1, lrf, 350)  # cosine 1->hyp['lrf']
        else:
            self.lf = (
                lambda x: max(1 - x / self.epochs, 0) * (1.0 - lrf) + lrf
            )  # linear
        return optim.lr_scheduler.LambdaLR(opt, lr_lambda=self.lf)

    def configure_optimizers(self):
        """Configures the optimizer and the scheduler."""
        # opt = torch.optim.SGD(self.modules.parameters(), lr=1e-2, weight_decay=0.0005)
        # opt = torch.optim.AdamW(
        #     self.modules.parameters(), lr=0.000119, weight_decay=0.0
        # )
        opt, lr = self.build_optimizer(self.modules, name="auto", lr=0.01, momentum=0.9)
        sched = self._setup_scheduler(opt, 0.01, lr)

        return opt, sched

    @torch.no_grad()
    def on_train_epoch_end(self):
        """
        Computes the mean average precision (mAP) at the end of the training epoch
        and logs the metrics in `metrics.txt` inside the experiment folder.
        The `verbose` argument if set to `True` prints details regarding the
        number of images, instances and metrics for each class of the dataset.
        The `plots` argument, if set to `True`, saves in the `runs/detect/train`
        folder the plots of the confusion matrix, the F1-Confidence,
        Precision-Confidence, Precision-Recall, Recall-Confidence curves and the
        predictions and labels of the first three batches of images.
        """
        return


def replace_datafolder(hparams, data_cfg):
    """Replaces the data root folder, if told to do so from the configuration."""
    print(data_cfg["train"])
    data_cfg["path"] = str(data_cfg["path"])
    data_cfg["path"] = (
        data_cfg["path"][:-1] if data_cfg["path"][-1] == "/" else data_cfg["path"]
    )
    for key in ["train", "val"]:
        if not isinstance(data_cfg[key], list):
            data_cfg[key] = [data_cfg[key]]
        new_list = []
        for tmp in data_cfg[key]:
            if hasattr(hparams, "data_dir"):
                if hparams.data_dir != data_cfg["path"]:
                    tmp = str(tmp).replace(data_cfg["path"], "")
                    tmp = tmp[1:] if tmp[0] == "/" else tmp
                    tmp = os.path.join(hparams.data_dir, tmp)
                    new_list.append(tmp)
        data_cfg[key] = new_list

    data_cfg["path"] = hparams.data_dir

    return data_cfg


if __name__ == "__main__":
    assert len(sys.argv) > 1, "Please pass the configuration file to the script."
    hparams = parse_configuration(sys.argv[1])
    if len(hparams.input_shape) != 3:
        hparams.input_shape = [
            int(x) for x in "".join(hparams.input_shape).split(",")
        ]  # temp solution
        print(f"Setting input shape to {hparams.input_shape}.")

    m_cfg, data_cfg = load_config(hparams.data_cfg)

    # check if specified path for images is different, correct it in case
    # data_cfg = replace_datafolder(hparams, data_cfg)
    m_cfg.imgsz = hparams.input_shape[-1]  # temp solution

    train_loader, val_loader = create_loaders(m_cfg, data_cfg, hparams.batch_size)

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(
        exp_folder, hparams=hparams, key="loss"
    )

    yolo_mind = YOLO(m_cfg, hparams=hparams)

    yolo_mind.train(
        epochs=hparams.epochs,
        datasets={"train": train_loader, "val": val_loader},
        metrics=[],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )
