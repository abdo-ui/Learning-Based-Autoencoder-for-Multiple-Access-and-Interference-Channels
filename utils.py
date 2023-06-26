import argparse
import datetime
import os
import random

import numpy as np
import torch
from torch import nn


def setup_seed(seed: int = 42) -> None:
    """Setup random seed for the environment to ensure reproducible results

    Parameters
    ----------
    seed : int, optional
        Random seed, by default 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    gen = torch.Generator()
    gen.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(
        prog="TrainAndTestModel",
        description="train and test the desired model with a given set of hyperparameters",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="Model name",
        type=str,
        default="proposed_model",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Configuration (Multiple Access \ Multi-User 'Interference Channel')",
        type=str,
        choices=("mac", "multiuser"),
        default="mac",
    )
    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=42)
    parser.add_argument(
        "-log", "--logRun", help="Enable Neptune logging", type=bool, default=False
    )
    parser.add_argument(
        "-e", "--epochs", help="Number of training epochs", type=int, default=30
    )
    parser.add_argument(
        "-lr", "--learningRate", help="Learning rate", type=float, default=0.0001
    )
    parser.add_argument("-k", "--k", help="Number of message bits", type=int, default=7)
    parser.add_argument("-L", "--L", help="Codeword Length", type=int, default=21)
    parser.add_argument(
        "-bs", "--batchSize", help="Number of samples per batch", type=int, default=32
    )

    parser.add_argument(
        "-A",
        "--trainingA",
        help="Signals peak intensity during training",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "-trn",
        "--trainNum",
        help="Number of training samples",
        type=int,
        default=10_000_000,
    )
    parser.add_argument(
        "-van",
        "--valNum",
        help="Number of validation samples",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-ten",
        "--testNum",
        help="Number testing of samples",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "-f", "--fading", help="Enable fading simulation", type=bool, default=False
    )
    parser.add_argument(
        "-fs",
        "--fadingSigma",
        help="Standard deviation of fading distribution (ignored if --fading is False)",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "-loss",
        "--lossType",
        help="How to combine the losses the two autoencoders",
        type=str,
        choices=("max", "combined"),
        default="max",
    )

    return parser.parse_args()


def generate_result_path(params):
    PATH = "results/"
    PATH += f"{params['name']}/"
    PATH += f"{params['k']}-{params['L']}/"
    PATH += f"fading/{params['fading_sigma']}/" if params["fading"] else "AWGN/"
    PATH += f"A{params['A']}/"

    current_date_time = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M")
    PATH += current_date_time + "/"

    os.makedirs(PATH, exist_ok=True)
    return PATH


class SaveBestModel:
    def __init__(
        self,
        best_valid_loss: float = float("inf"),
        path: str = ".",
        name: str = "best_model",
    ) -> None:
        self.best_valid_loss = best_valid_loss
        self.file_path = f"{path}/model_weights/{name}.pth"
        os.makedirs(f"{path}/model_weights", exist_ok=True)

    def __call__(
        self,
        current_valid_loss: float,
        epoch: int,
        model: nn.Module,
    ) -> None:
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(
                f"    Best validation loss: {self.best_valid_loss} - Saving best model for epoch: {epoch}"
            )
            torch.save(model.state_dict(), self.file_path)
