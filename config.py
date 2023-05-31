from typing import Union
from dataclasses import dataclass


@dataclass
class Config:
    "Config for the VAE MNIST Example by pytorch."
    batch_size: int = 128
    epochs: int = 7
    # Must use Union here and not the new | syntax
    gpu: Union[int, None] = None
    no_mps: bool = False
    seed: int = 1
    eval_interval: int = 2
    wandb: bool = True

    # Some stuff for filtering over
    weight_decay: float = 0.0
    lr: float = 1e-3
