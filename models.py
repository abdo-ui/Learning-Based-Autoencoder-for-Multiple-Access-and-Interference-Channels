from abc import ABC, abstractmethod
from typing import Tuple, Literal

import torch
from torch import Tensor, nn


class RTNBase(ABC, nn.Module):
    def __init__(
        self,
        in_channels: int,
        compressed_dim: int,
        channel_mode: Literal["mac", "multiuser"] = "mac",
        fading: bool = False,
        f_sigma: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.compressed_dim = compressed_dim
        self.fading = fading
        self.f_sigma = f_sigma

        self.channel_mode = channel_mode
        if self.channel_mode not in ["mac", "multiuser"]:
            raise NotImplementedError(
                f"Channel mode '{self.channel_mode}' is not implemented"
            )

        self.encoder1 = self.build_encoder(self.in_channels, self.compressed_dim)
        self.encoder2 = self.build_encoder(self.in_channels, self.compressed_dim)

        self.decoder1 = self.build_decoder(self.compressed_dim, self.in_channels)
        self.decoder2 = self.build_decoder(self.compressed_dim, self.in_channels)

    @abstractmethod
    def build_encoder(self, input_shape: int, output_shape: int):
        pass

    @abstractmethod
    def build_decoder(self, input_shape: int, output_shape: int):
        pass

    def mixedAWGN(self, x1: Tensor, x2: Tensor, A: float) -> Tuple[Tensor, Tensor]:
        x1 = torch.sigmoid(x1) * A
        x2 = torch.sigmoid(x2) * A

        if self.channel_mode == "mac":
            return self.apply_MAC_channel(x1, x2)
        elif self.channel_mode == "multiuser":
            return self.apply_multiuser_channel(x1, x2)

    def apply_MAC_channel(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
        x_channel = x1 + x2
        noise = torch.randn(x_channel.size())

        if self.fading:
            m3 = -self.f_sigma**2 / 2
            f_true = torch.empty_like(x_channel).log_normal_(m3, self.f_sigma)
            signal1 = signal2 = x_channel + (noise / f_true)
        else:
            signal1 = signal2 = x_channel + noise

        return signal1, signal2

    def apply_multiuser_channel(self, x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
        noise1 = torch.randn(x1.size())
        noise2 = torch.randn(x2.size())

        if self.fading:
            m3 = -self.f_sigma**2 / 2
            f1 = torch.empty_like(x1).log_normal_(m3, self.f_sigma)
            f2 = torch.empty_like(x2).log_normal_(m3, self.f_sigma)
            f_avg = (f1 + f2) / 2

            signal1 = x1 * f1 + x2 * f2 + noise1
            signal1 /= f_avg

            signal2 = x1 * f1 + x2 * f2 + noise2
            signal2 /= f_avg

        else:
            signal1 = x1 + x2 + noise1
            signal2 = x1 + x2 + noise2

        return signal1, signal2

    def get_constellation_points(
        self, x1: Tensor, x2: Tensor, A: float
    ) -> Tuple[Tensor, Tensor]:
        encoded1 = torch.sigmoid(self.encoder1(x1)) * A
        encoded2 = torch.sigmoid(self.encoder2(x2)) * A
        return encoded1, encoded2

    def forward(self, x1: Tensor, x2: Tensor, A: float = 2.9) -> tuple[Tensor, Tensor]:
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)

        signal1, signal2 = self.mixedAWGN(x1, x2, A)

        x1_decoded = self.decoder1(signal1)
        x2_decoded = self.decoder2(signal2)

        return x1_decoded, x2_decoded


class RTNMAC(RTNBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if kwargs["channel_mode"] != "mac":
            raise ValueError(
                f"channel mode in RTNNormMAC must be 'mac', got '{kwargs['channel_mode']}' instead."
            )

        self.decoder_input_normalizer = nn.LayerNorm(self.compressed_dim)

    def build_encoder(
        self, input_shape: int = 128, output_shape: int = 21
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.Linear(100, output_shape),
            nn.BatchNorm1d(output_shape),
        )

    def build_decoder(
        self, input_shape: int = 21, output_shape: int = 128
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.LayerNorm(100),
            nn.Linear(100, output_shape),
            nn.LayerNorm(output_shape),
        )

    def forward(self, x1: Tensor, x2: Tensor, A: float = 2.9) -> Tuple[Tensor, Tensor]:
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)

        signal1, signal2 = self.mixedAWGN(x1, x2, A)
        normalized_signal1 = self.decoder_input_normalizer(signal1)
        normalized_signal2 = self.decoder_input_normalizer(signal2)

        x1_decoded = self.decoder1(normalized_signal1)
        x2_decoded = self.decoder2(normalized_signal2)

        return x1_decoded, x2_decoded


class RTNMU(RTNBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if kwargs["channel_mode"] != "multiuser":
            raise ValueError(
                f"channel mode in RTNNormMU must be 'multiuser', got '{kwargs['channel_mode']}' instead."
            )

    def build_encoder(
        self, input_shape: int = 128, output_shape: int = 21
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.BatchNorm1d(100),
            nn.Linear(100, output_shape),
            nn.BatchNorm1d(output_shape),
        )

    def build_decoder(
        self, input_shape: int = 21, output_shape: int = 128
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_shape),
            nn.Linear(input_shape, 100),
            nn.LayerNorm(100),
            nn.Linear(100, output_shape),
            nn.LayerNorm(output_shape),
        )
