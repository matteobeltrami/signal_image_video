from typing import Dict, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm
import os

from torch.cuda import device
import torch.nn as nn
import torch

from .utils.helpers import select_and_load_checkpoint
from .utils.checkpointer import Checkpointer

@dataclass
class Stage:
    train: int = 0
    val: int = 1
    test: int = 2

class MicroMind(ABC):
    def __init__(self):
        # here we should handle devices etc.
        self.modules = torch.nn.ModuleDict({}) # init empty modules dict
        self.input_shape = None

        self.device = "cpu"
        self.accelerator = Accelerator()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def compute_loss(self, pred, batch):
        pass

    def set_input_shape(self, input_shape: Tuple = (3, 224, 224)):
        self.input_shape = input_shape

    def load_modules(self, checkpoint_path: Union[Path, str]):
        """ Loads models for path. """
        dat = torch.load(checkpoint_path)

        modules_keys = list(self.modules.keys())
        for k in self.modules:
            self.modules[k].load_state_dict(dat[k])

            modules_keys.remove(k)

        if len(modules_keys) != 0:
            print(modules_keys)
            breakpoint()
            logger.info(f"Couldn't find a state_dict for modules {modules_keys}.")

        def export(self, out_format: str = "onnx"):
            pass
            # from micromind import convert
# 
            # if out_format == "onnx":
                # convert.convert_to_onnx(self)

    def export(self, save_dir: Union[Path, str], out_format: str = "onnx", input_shape=None):
        from micromind import convert
        if not isinstance(save_dir, Path): save_dir = Path(save_dir)

        self.set_input_shape(input_shape)
        assert self.input_shape is not None, "You should pass the input_shape of the model."

        if out_format == "onnx":
            convert.convert_to_onnx(self, save_dir.joinpath("model.onnx"))
        elif out_format == "openvino":
            convert.convert_to_openvino(self, save_dir)
        elif out_format == "tflite":
            convert.convert_to_tflite(self, save_dir)

    def configure_optimizers(self):
        opt_conf = {"lr": 0.001}
        opt = torch.optim.Adam(self.modules.parameters(), **opt_conf)
        return opt, None    # None is for learning rate sched

    def __call__(self, *x, **xv):
        return self.forward(*x, **xv)

    def on_train_start(self):
        # this should be loaded from argparse
        self.output_folder = "results"
        self.experiment_name = "test01"
        self.experiment_folder = os.path.join(
            self.output_folder,
            self.experiment_name
        )

        save_dir = os.path.join(
                self.experiment_folder, "save"
            )
        if os.path.exists(save_dir):
            if len(os.listdir(save_dir)) != 0:
                # select which checkpoint and load it.
                checkpoint, path = select_and_load_checkpoint(save_dir)
                self.opt = checkpoint["optimizer"]
                self.lr_sched = checkpoint["lr_scheduler"]
                self.start_epoch = checkpoint["epoch"] + 1

                self.load_modules(path)

                if self.accelerator.is_local_main_process:
                    self.checkpointer = Checkpointer(
                        checkpoint["key"],
                        mode=checkpoint["mode"],
                        checkpoint_path=self.experiment_folder
                    )

                    logger.info(f"Loaded existing checkpoint from {path}.")
            else:
                self.opt, self.lr_sched = self.configure_optimizers()
                self.start_epoch = 0
    
                self.checkpointer = Checkpointer("loss", checkpoint_path=self.experiment_folder)
        else:
            os.makedirs(self.experiment_folder, exist_ok=True)

            self.opt, self.lr_sched = self.configure_optimizers()
            self.start_epoch = 0

            self.checkpointer = Checkpointer("loss", checkpoint_path=self.experiment_folder)


        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.modules.to(self.device)
        print("Set device to ", self.device)

        convert = [self.modules, self.opt, self.lr_sched] + list(self.datasets.values())
        accelerated = self.accelerator.prepare(convert)
        self.modules, self.opt, self.lr_sched = accelerated[:3]
        for i, key in enumerate(self.datasets):
            self.datasets[key] = accelerated[-(i + 1)]

    def on_train_end(self):
        if self.accelerator.is_local_main_process:
            self.checkpointer.close()

    def train(
            self,
            epochs: int = 1,
            datasets: Dict = {},
            debug: bool = False
        ) -> None:
        self.datasets = datasets
        assert "train" in self.datasets, "Training dataloader was not specified."
        assert epochs > 0, "You must specify at least one epoch."

        self.debug = debug

        self.on_train_start()

        if self.accelerator.is_local_main_process:
            logger.info(f"Starting from epoch {self.start_epoch}. Training is scheduled for {epochs} epochs.")
        with self.accelerator.autocast():
            for e in range(self.start_epoch, epochs):
                pbar = tqdm(self.datasets["train"], unit="batches", ascii=True, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process)
                loss_epoch = 0
                pbar.set_description(f"Running epoch {e + 1}/{epochs}")
                self.modules.train()
                for idx, batch in enumerate(pbar):
                    self.opt.zero_grad()
        
                    loss = self.compute_loss(self(batch), batch)
    
                    self.accelerator.backward(loss)
                    self.opt.step()
        
                    loss_epoch += loss.item()
                    pbar.set_postfix(loss=loss_epoch/(idx + 1))
        
                    if self.debug and idx > 10: break
    
                pbar.close()
    
                if "val" in datasets: self.validate()
                if self.accelerator.is_local_main_process:
                    self.checkpointer(self, e, self.val_metrics, lambda x: self.accelerator.unwrap_model(x))
    
                if e >= 1 and self.debug: break


        self.on_train_end()
        return None

    @torch.no_grad()
    def validate(self) -> None:
        assert "val" in self.datasets, "Validation dataloader was not specified."
        self.modules.eval()

        pbar = tqdm(self.datasets["val"], unit="batches", ascii=True, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process)
        loss_epoch = 0
        pbar.set_description(f"Validation...")
        with self.accelerator.autocast():
            for idx, batch in enumerate(pbar):
                self.opt.zero_grad()
    
                loss = self.compute_loss(self(batch), batch)
    
                loss_epoch += loss.item()
                pbar.set_postfix(loss=loss_epoch/(idx + 1))
    
                if self.debug and idx > 10: break
    
        self.val_metrics = {"loss": loss_epoch / (idx + 1)}

        pbar.close()

        return None

    @torch.no_grad()
    def test(self, datasets: Dict = {}) -> None:
        assert "test" in self.datasets, "Test dataloader was not specified."
        self.modules.eval()

        pbar = tqdm(self.datasets["test"], unit="batches", ascii=True, dynamic_ncols=True, disable=not self.accelerator.is_local_main_process)
        loss_epoch = 0
        pbar.set_description(f"Testing...")
        with self.accelerator.autocast():
            for idx, batch in enumerate(pbar):
                self.opt.zero_grad()
    
                loss = self.compute_loss(self(batch), batch)
    
                loss_epoch += loss.item()
                pbar.set_postfix(loss=loss_epoch/(idx + 1))

        pbar.close()

        return None

