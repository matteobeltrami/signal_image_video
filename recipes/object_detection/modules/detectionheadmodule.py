# Ultralytics YOLO custom model DetectionHeadModule.py

import contextlib
from pathlib import Path

import torch
import torch.nn as nn

from micromind import PhiNet

from ultralytics.nn.tasks import DetectionModel

from ultralytics.nn.modules import (
    Detect,
    Pose,
    Segment,
)
from ultralytics.yolo.utils import (
    LOGGER,
    yaml_load,
)
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.torch_utils import initialize_weights

from modules.microhead import Microhead

try:
    import thop
except ImportError:
    thop = None


class DetectionHeadModel(DetectionModel):
    """YOLOv8 detection model."""

    def __init__(
        self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True
    ):  # model, input channels, number of classes

        super(DetectionModel, self).__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Read custom hardcoded model
        self.model, self.save = parse_model_custom(
            nc, ch=ch, verbose=verbose
        )  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = (
                lambda x: self.forward(x)[0]
                if isinstance(m, (Segment, Pose))
                else self.forward(x)
            )
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))]
            )  # forward
            self.stride = m.stride
            m.bias_init()  # only run once

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(
            f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix."
            f"Renaming {path.stem} to {new_stem}."
        )
        path = path.with_stem(new_stem)

    unified_path = re.sub(
        r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path)
    )  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size
    character of the model's scale. The function uses regular expression matching
    to find the pattern of the model scale in the YAML file name, which is denoted
    by n, s, m, l, or x. The function returns the size character of the model
    scale as a string.

    Args:
        model_path (str) or (Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re

        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(
            1
        )  # n, s, m, l, or x
    return ""


def get_output_dim(data_config, model):
    """Returns intermediate representations shapes."""
    x = torch.randn(*[1] + list(data_config["input_size"]))
    out_dim = [model._layers[0](x)]
    names = [model._layers[0].__class__]
    for layer in model._layers[1:]:
        out_dim.append(layer(out_dim[-1]))
        names.append(layer.__class__)
    return [list(o.shape)[1:] for o in out_dim], names


def get_output_dim_layers(data_config, layers):
    x = torch.randn(*[1] + list(data_config["input_size"]))
    out_dim = [layers[0](x)]
    names = [layers[0].__class__]
    for layer in layers[1:]:
        out_dim.append(layer(out_dim[-1]))
        names.append(layer.__class__)
    return [list(o.shape)[1:] for o in out_dim], names


def parse_model_custom(nc, ch, verbose=True):

    if verbose:
        LOGGER.info(
            f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}"
            f"  {'module':<45}{'dimensions':<30}"
        )
    ch = [ch]
    layers, save = [], []  # layers, savelist, ch out

    # START layers phinet ------------------------------------
    model = PhiNet(
        input_shape=(3, 320, 320),
        alpha=2.67,
        num_layers=6,
        beta=1,
        t_zero=4,
        include_top=False,
        num_classes=nc,
        compatibility=False,
    )
    layers = list(model._layers)

    # END layers phinet ------------------------------------
    #
    # maybe we can take away this code?
    data_config = {"input_size": (3, 320, 320)}
    res = get_output_dim(data_config, model)

    for i, layer in enumerate(model._layers):
        f = -1
        n_ = 1
        args = res[0][i]
        t = str(layer.__class__).replace("<class '", "").replace("'>", "")
        layer.np = sum(x.numel() for x in layer.parameters())  # number params
        layer.i, layer.f, layer.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(
                f"{i:>3}{str(f):>20}{n_:>3}{layer.np:10.0f}  {t:<45}{str(args):<30}"
            )  # print

    # START HARDCODED detection head ------------------------------------

    module_head = Microhead()
    layers += list(module_head._layers)
    save = module_head._save

    # END HARDCODED HEAD ------------------------------------------------------------

    return nn.Sequential(*layers), sorted(save)
