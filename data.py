import torch
from torch import Tensor

# Data Loader
class CustomDataLoader:
    def __init__(
        self,
        *tensors: tuple[Tensor, ...],
        batch_size: int = None,
        shuffle: bool = False,
    ) -> None:
        self.tensors = tensors
        self.n_samples = tensors[0].shape[0]
        self.batch_size = batch_size if batch_size is not None else self.n_samples
        self.shuffle = shuffle

        self.n_batches, remainder = divmod(self.n_samples, self.batch_size)
        self.n_batches += remainder > 0
        self.indices = torch.arange(self.n_samples)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> "CustomDataLoader":
        if self.shuffle:
            self.indices = torch.randperm(self.n_samples)
        self.i = 0
        return self

    def __next__(self) -> tuple[Tensor, ...]:
        if self.i >= self.n_samples:
            raise StopIteration

        batch_indices = self.indices[self.i : self.i + self.batch_size]
        batch = tuple(t[batch_indices] for t in self.tensors)
        self.i += self.batch_size
        return batch


def generate_data(
    n_samples: int = 1_000_000, n_dimensions: int = 128
) -> tuple[Tensor, Tensor]:
    """Generate One-hot-encoded samples and their corresponding labels.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples in the generated data, by default 1_000_000.
    n_dimensions : int, optional
        Number of dimension in the one-hot-encoded data, by default 128.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing the data and labels
    """
    labels = torch.randint(n_dimensions, (n_samples,))
    data = torch.eye(n_dimensions).index_select(dim=0, index=labels)

    return data, labels
