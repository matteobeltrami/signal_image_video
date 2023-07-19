from .networks.phinet import PhiNet
from .utils import configlib

try:
    from .yolo.model import microYOLO
    from .yolo.microyoloheadphiblock import Microhead
    from .yolo.utils.benchmarks import benchmark
    from .yolo.detection.detectionmicromodule import DetectionMicroModel
    from .yolo.detection.detectionmicrotrainer import DetectionMicroTrainer
except ImportError:
    print(
        "Warning: ultralytics.yolo package not found. microYOLO class not available."
        "-- pleas run the command 'pip install ultralytics' to install it."
    )


# Package version
__version__ = "0.0.4"

"""datasets_info is a dictionary that contains information about the attributes
of the datasets.
This dictionary is used in networks.py inside the from_pretrained class method
in order to examine the inputs and initialize the PhiNet or, in case of
mismatching between dataset and Nclasses, raise an AssertionError."""
datasets_info = {
    "CIFAR-100": {"Nclasses": 100, "NChannels": 3, "ext": ".pth.tar"},
    "CIFAR-10": {"Nclasses": 10, "NChannels": 3, "ext": ".pth.tar"},
    "ImageNet-1k": {"Nclasses": 1000, "NChannels": 3, "ext": ".pth.tar"},
    "MNIST": {"Nclasses": 10, "NChannels": 1, "ext": ".pth.tar"},
}
