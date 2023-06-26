import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from data import CustomDataLoader


def calculate_BLER(
    model, data_loader: CustomDataLoader, A: float
) -> tuple[float, float]:
    """Calculate the Block Error Rate (BLER) over a batch of data at a given signal amplitude

    Parameters
    ----------
    model : nn.Module
        The selected model
    data_loader : CustomDataLoader
        A loader object that generates batches of data. The batch should be a tuple of tensor with this order (x1, y1, x2, y2)
    A : float
        Signals amplitude.

    Returns
    -------
    tuple[float, float]
        BLER for both signal_1 and signal_2.
    """
    BLER1 = BLER2 = 0
    model.eval()
    with torch.no_grad():
        for x1, y1, x2, y2 in data_loader:
            x1_decoded, x2_decoded = model(x1, x2, A)

            y1_pred = torch.argmax(x1_decoded, axis=1)
            y2_pred = torch.argmax(x2_decoded, axis=1)

            BLER1 += torch.ne(y1, y1_pred).sum().float()
            BLER2 += torch.ne(y2, y2_pred).sum().float()

        BLER1 /= data_loader.n_samples
        BLER2 /= data_loader.n_samples

        return BLER1, BLER2


def calculate_BER_from_BLER(BLER: float, k: int) -> float:
    """Calculate the Bit Error Rate (BER) from the Block Error Rate (BLER) at a given k

    Parameters
    ----------
    BLER : float
        Block Error Rate
    k : int
        K

    Returns
    -------
    BER: float
        Bit Error Rate
    """
    BER = 1 - (1 - BLER) ** (1 / k)
    return BER


def plot_BER(
    EpNo: list[float],
    BER: list[float],
    label: str = "Model",
):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(EpNo, BER, "bo-", label=label)

    ax.set_title("Average Bit Error Rate", fontsize=15)
    ax.set_xlabel("EpNo (dB)")
    ax.set_ylabel("BER")

    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--")
    plt.legend(loc="upper right", ncol=1)
    plt.xticks(np.arange(-6, max(EpNo)))

    return fig


def plot_constellation_hist(const_pts, n_bins: int = 50) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    hist = sns.histplot(const_pts, bins=n_bins, stat="proportion", ax=ax)
    hist.set(title="Constellation points distribution")

    return fig
