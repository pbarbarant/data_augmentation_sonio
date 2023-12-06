# %%
from typing import Optional

import torch
import torch.nn as nn

from pythae.models import VAE, VAEConfig
from pythae.data.datasets import BaseDataset
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pydantic.dataclasses import dataclass


@dataclass
class CVAEConfig(VAEConfig):
    """Configuration class for CVAE model"""

    channels: int = 3
    image_height: int = 128
    image_width: int = 128
    class_size: int = 3
    latent_dim: int = 10

    def __post_init__(self):
        super().__post_init__()
        self.input_dim = (
            self.channels * self.image_height * self.image_width + self.class_size
        )


class Custom_encoder(BaseEncoder):
    """Encoder for CVAE model

    Parameters
    ----------
    args : ModelConfig
        Configuration class for CVAE model
    """

    def __init__(self, args=None):
        BaseEncoder.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(args.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, args.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = self.layers(x)
        log_covariance = out
        output = ModelOutput(embedding=out, log_covariance=log_covariance)
        return output


class Custom_decoder(BaseDecoder):
    """Decoder for CVAE model

    Parameters
    ----------
    args : ModelConfig
        Configuration class for CVAE model
    """

    def __init__(self, args=None):
        BaseDecoder.__init__(self)
        self.class_size = args.class_size
        self.layers = nn.Sequential(
            nn.Linear(args.latent_dim + args.class_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, args.input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = self.layers(x)
        log_covariance = torch.zeros_like(out)
        output = ModelOutput(
            reconstruction=out[
                :, : -self.class_size
            ],  # Set the output from the decoder in a ModelOutput instance
            log_covariance=log_covariance,
        )
        return output


class CVAE(VAE):
    def __init__(
        self,
        model_config: VAEConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        super().__init__(model_config, encoder, decoder)
        self.model_name = "CVAE"
        self.input_dim = model_config.input_dim
        self.class_size = model_config.class_size
        self.encoder = encoder
        self.decoder = decoder

    def conditional_encoder(self, x, y):
        inputs = torch.cat([x, y], 1)  # (bs, feature_size+class_size)
        return self.encoder(inputs)

    def conditional_decoder(self, z, y):
        inputs = torch.cat([z, y], 1)  # (bs, latent_size+class_size)
        return self.decoder(inputs)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        Forward pass of the model

        Args:
            inputs (BaseDataset): input data
            **kwargs: additional arguments

        Returns:
            ModelOutput: model outputs
        """

        x = inputs.data  # image
        y = inputs.labels  # one-hot encoding

        encoder_output = self.conditional_encoder(x, y)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)

        recon_x = self.conditional_decoder(z, y)["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output


if __name__ == "__main__":
    cvae_config = CVAEConfig(
        latent_dim=10,
        class_size=3,
    )

    # Generate random data
    n_samples = 100
    x = torch.randn(n_samples, 3, 128, 128).view(n_samples, -1)
    y = torch.randn(n_samples, 3)
    inputs = BaseDataset(data=x, labels=y)

    encoder = Custom_encoder(cvae_config)
    decoder = Custom_decoder(cvae_config)

    # Test encoder
    encoder_input = torch.cat([x, y], 1)
    encoder_output = encoder(encoder_input)["embedding"]
    decoder_input = torch.cat([encoder_output, y], 1)
    decoder_output = decoder(decoder_input)["reconstruction"]
    print(decoder_output.shape)

    # Test forward pass
    model = CVAE(cvae_config, encoder, decoder)
    model_output = model(inputs)
    print(model_output.keys())
