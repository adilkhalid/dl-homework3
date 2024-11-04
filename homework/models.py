from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)

        self.upconv = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Forward pass through strided convolutions
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Upsample with Transposed Convolution
        x = self.upconv(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)


import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x (W*H) x C
        key = self.key_conv(x).view(batch_size, -1, width * height)  # B x C x (W*H)
        energy = torch.bmm(query, key)  # B x (W*H) x (W*H)
        attention = torch.softmax(energy, dim=-1)  # B x (W*H) x (W*H)
        value = self.value_conv(x).view(batch_size, -1, width * height)  # B x C x (W*H)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x  # Residual connection
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2  # Maintain spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class Detector(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(Detector, self).__init__()

        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.tensor(INPUT_STD))

        # Downsampling path with 4 layers
        self.down1 = ConvBlock(in_channels, 16, stride=2)  # (B, 16, H/2, W/2)
        self.down2 = ConvBlock(16, 32, stride=2)  # (B, 32, H/4, W/4)
        self.down3 = ConvBlock(32, 64, stride=2)  # (B, 64, H/8, W/8)
        self.down4 = ConvBlock(64, 128, stride=2)  # (B, 128, H/16, W/16)

        # Bottleneck layer
        self.bottleneck = ConvBlock(128, 256)

        # Upsampling path with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),  # Input is 256 (128 + 128) after concatenation
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),  # Input is 128 (64 + 64) after concatenation
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2),  # Input is 64 (32 + 32) after concatenation
            nn.ReLU()
        )

        # **Add this extra upsampling layer**
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )

        # Update the output heads to match the number of input channels
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Normalize input
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Downsampling path
        d1 = self.down1(x)  # (B, 16, H/2, W/2)
        d2 = self.down2(d1)  # (B, 32, H/4, W/4)
        d3 = self.down3(d2)  # (B, 64, H/8, W/8)
        d4 = self.down4(d3)  # (B, 128, H/16, W/16)

        # Bottleneck
        bottleneck = self.bottleneck(d4)  # (B, 256, H/16, W/16)

        # Upsampling path
        u1 = self.up1(bottleneck)
        u1 = torch.cat([u1, d4], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)

        # **Apply the extra upsampling layer**
        u5 = self.up5(u4)

        # Output layers
        logits = self.segmentation_head(u5)  # (B, num_classes, H, W)
        depth = self.depth_head(u5)  # (B, 1, H, W)

        return logits, depth

    def predict(self, x):
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)
        return pred, raw_depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
        model_name: str,
        with_weights: bool = False,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
