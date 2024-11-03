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




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2  # Standard padding to maintain spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class Detector(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(Detector, self).__init__()

        # Register buffers for input normalization
        self.register_buffer("input_mean", torch.tensor([0.2788, 0.2657, 0.2629]))
        self.register_buffer("input_std", torch.tensor([0.2064, 0.1944, 0.2252]))

        # Downsampling path
        self.down1 = ConvBlock(in_channels, 16, stride=2)  # Down1: (B, 16, H/2, W/2)
        self.down2 = ConvBlock(16, 32, stride=2)           # Down2: (B, 32, H/4, W/4)

        # Upsampling path with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)  # ReLU after up-convolution
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)  # ReLU after up-convolution
        )

        # Output heads for segmentation and depth
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)  # Logits: (B, num_classes, H, W)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)                   # Depth: (B, 1, H, W)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Normalize input
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Downsampling
        d1 = self.down1(x)  # (B, 16, H/2, W/2)
        d2 = self.down2(d1)  # (B, 32, H/4, W/4)

        # Upsampling with skip connections
        u1 = self.up1(d2)  # (B, 16, H/2, W/2)
        u1 = torch.cat((u1, d1), dim=1)  # Concatenate with d1

        u2 = self.up2(u1)  # (B, 16, H, W)

        # Output layers
        logits = self.segmentation_head(u2)  # (B, num_classes, H, W)
        depth = torch.sigmoid(self.depth_head(u2))  # (B, 1, H, W)

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
