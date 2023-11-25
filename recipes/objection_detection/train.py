"""
YOLO training.

Authors:
    - Matteo Beltrami, 2023
    - Francesco Paissan, 2023

This code allows you to train an object detection model with the YOLOv8 neck and loss.

To run this script, you can start it with:
    python train.py
"""

import torch
from prepare_data import create_loaders
from torchinfo import summary
from ultralytics.utils.ops import scale_boxes, xywh2xyxy
from huggingface_hub import hf_hub_download

from yolo_loss import Loss

import micromind as mm
from micromind.networks import PhiNet
from micromind.networks.yolo import SPPF, DetectionHead, Yolov8Neck
from micromind.utils.parse import parse_arguments
from micromind.utils.yolo import (
    load_config,
    mean_average_precision,
    postprocess,
)
import sys


class YOLO(mm.MicroMind):
    def __init__(self, m_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        REPO_ID = "micromind/ImageNet"
        FILENAME = "v5/state_dict.pth.tar"
        
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        args = Path(FILENAME).parent.joinpath("args.yaml")
        with open(args, "r") as f:
            dat = yaml.safe_load(f)

        input_shape = (3, 672, 672)
        alpha = dat["alpha"]
        num_layers = dat["num_layers"]
        beta = dat["beta"]
        t_zero = dat["t_zero"]
        divisor = 8
        downsampling_layers = [4, 5, 7]
        return_layers = [5, 6, 7]

        self.modules["phinet"] = PhiNet(
            input_shape=input_shape,
            alpha=alpha,
            num_layers=num_layers,
            beta=beta,
            t_zero=t_zero,
            include_top=False,
            compatibility=False,
            divisor=divisor,
            downsampling_layers=downsampling_layers,
            return_layers=return_layers,
        )

        # load ImageNet checkpoint
        self.modules["phinet"].load_state_dict(torch.load(model_path), strict=False)

        sppf_ch, neck_filters, up, head_filters = self.get_parameters()

        self.modules["sppf"] = SPPF(*sppf_ch)
        self.modules["neck"] = Yolov8Neck(filters=neck_filters, up=up)
        self.modules["head"] = DetectionHead(filters=head_filters)

        tot_params = 0
        for m in self.modules.values():
            temp = summary(m, verbose=0)
            tot_params += temp.total_params

        print(f"Total parameters of model: {tot_params * 1e-6:.2f} M")

        self.m_cfg = m_cfg

    def get_parameters(self):
        """
        Gets the parameters with which to initialize the network detection part
        (SPPF block, Yolov8Neck, DetectionHead).
        """
        in_shape = self.modules["phinet"].input_shape
        x = torch.randn(1, *in_shape)
        y = self.modules["phinet"](x)

        c1 = c2 = y[1][2].shape[1]
        sppf = SPPF(c1, c2)
        out_sppf = sppf(y[1][2])

        neck_filters = [y[1][0].shape[1], y[1][1].shape[1], out_sppf.shape[1]]
        up = [2, 2]
        up[0] = y[1][1].shape[2] / out_sppf.shape[2]
        up[1] = y[1][0].shape[2] / (up[0] * out_sppf.shape[2])
        temp = """The layers you selected are not valid. \
            Please choose only layers between which the spatial resolution \
            doubles every time. You can try changing the downsampling layers"""

        assert up == [2, 2], " ".join(temp.split())

        neck = Yolov8Neck(filters=neck_filters, up=up)
        out_neck = neck(y[1][0], y[1][1], out_sppf)

        head_filters = (
            out_neck[0].shape[1],
            out_neck[1].shape[1],
            out_neck[2].shape[1],
        )
        # head = DetectionHead(filters=head_filters)

        return (c1, c2), neck_filters, up, head_filters

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        preprocessed_batch = {}
        preprocessed_batch["img"] = (
            batch["img"].to(self.device, non_blocking=True).float() / 255
        )
        for k in batch:
            if isinstance(batch[k], torch.Tensor) and k != "img":
                preprocessed_batch[k] = batch[k].to(self.device)

        return preprocessed_batch

    def forward(self, batch):
        preprocessed_batch = self.preprocess_batch(batch)
        backbone = self.modules["phinet"](preprocessed_batch["img"].to(self.device))[1]
        backbone[-1] = self.modules["sppf"](backbone[-1])
        neck = self.modules["neck"](*backbone)
        head = self.modules["head"](neck)

        return head

    def compute_loss(self, pred, batch):
        self.criterion = Loss(self.m_cfg, self.modules["head"], self.device)
        preprocessed_batch = self.preprocess_batch(batch)

        lossi_sum, lossi = self.criterion(
            pred[1],
            preprocessed_batch,
        )

        return lossi_sum

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.modules.parameters(), lr=1e-2, beta1=0.937, weight_decay=0.0005)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            "min",
            factor=0.2,
            patience=50,
            threshold=10,
            min_lr=0,
            verbose=True,
        )
        return opt, sched

    @torch.no_grad()
    def mAP(self, pred, batch):
        preprocessed_batch = self.preprocess_batch(batch)
        post_predictions = postprocess(
            preds=pred[0], img=preprocessed_batch, orig_imgs=batch
        )

        batch_bboxes_xyxy = xywh2xyxy(batch["bboxes"])
        dim = batch["resized_shape"][0][0]
        batch_bboxes_xyxy[:, :4] *= dim

        batch_bboxes = []
        for i in range(len(batch["batch_idx"])):
            for b in range(len(batch_bboxes_xyxy[batch["batch_idx"] == i, :])):
                batch_bboxes.append(
                    scale_boxes(
                        batch["resized_shape"][i],
                        batch_bboxes_xyxy[batch["batch_idx"] == i, :][b],
                        batch["ori_shape"][i],
                    )
                )
        batch_bboxes = torch.stack(batch_bboxes)
        mmAP = mean_average_precision(post_predictions, batch, batch_bboxes)

        return torch.Tensor([mmAP])


if __name__ == "__main__":
    batch_size = 8
    hparams = parse_arguments()

    dset = input("Enter dataset configuration file path [Press Enter for COCO]: ")
    if dset == '': dset = "cfg/coco.yaml"
    m_cfg, data_cfg = load_config(dset)
    train_loader, val_loader = create_loaders(m_cfg, data_cfg, batch_size)

    exp_folder = mm.utils.checkpointer.create_experiment_folder(
        hparams.output_folder, hparams.experiment_name
    )

    checkpointer = mm.utils.checkpointer.Checkpointer(exp_folder, key="loss")

    yolo_mind = YOLO(m_cfg, hparams=hparams)

    mAP = mm.Metric("mAP", yolo_mind.mAP, eval_only=True, eval_period=1)

    yolo_mind.train(
        epochs=20,
        datasets={"train": train_loader, "val": val_loader},
        metrics=[mAP],
        checkpointer=checkpointer,
        debug=hparams.debug,
    )
