import os
import sys
import numpy as np
import torch
import torch.nn as nn


# ==========================================
# NAFNET
# ==========================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + 1e-6)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()

        dw_channel = c * dw_expand

        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(
            dw_channel, dw_channel, 3, 1, 1,
            groups=dw_channel
        )
        self.sg = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2,
                dw_channel // 2,
                1, 1, 0
            )
        )

        self.conv3 = nn.Conv2d(
            dw_channel // 2, c, 1, 1, 0
        )

        self.norm2 = LayerNorm2d(c)

        ffn_channel = ffn_expand * c

        self.conv4 = nn.Conv2d(
            c, ffn_channel, 1, 1, 0
        )

        self.conv5 = nn.Conv2d(
            ffn_channel // 2, c, 1, 1, 0
        )

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):

        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)

        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)

        x = x + y * self.gamma

        return x


class NAFNet(nn.Module):

    def __init__(
        self,
        img_channel=1,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1]
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            img_channel, width, 3, 1, 1
        )

        self.ending = nn.Conv2d(
            width, img_channel, 3, 1, 1
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width

        for num in enc_blk_nums:

            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

            self.downs.append(
                nn.Conv2d(chan, chan * 2, 2, 2)
            )

            chan *= 2

        self.middle = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num in dec_blk_nums:

            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, 1, 0),
                    nn.PixelShuffle(2)
                )
            )

            chan //= 2

            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

    def forward(self, x):

        inp = x
        x = self.intro(x)

        encs = []

        for encoder, down in zip(
            self.encoders,
            self.downs
        ):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, enc_skip in zip(
            self.decoders,
            self.ups,
            encs[::-1]
        ):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        return x + inp


class NAFNetWithSR(nn.Module):

    def __init__(self, nafnet):
        super().__init__()

        self.nafnet = nafnet

        self.upscale = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.upscale(self.nafnet(x))


# ==========================================
# MAIN
# ==========================================

def main():

    if len(sys.argv) != 3:
        print("Usage:")
        print("python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Model path
    model_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "nafnet_final.pth"
    )

    if not os.path.exists(model_path):
        print("ERROR: Model weights not found:")
        print(model_path)
        sys.exit(1)

    # Create model
    nafnet = NAFNet(
        img_channel=1,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1]
    )

    model = NAFNetWithSR(nafnet)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Find .npy files
    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".npy")
    ])

    print("Input files:", len(files))

    for filename in files:

        input_path = os.path.join(
            input_dir,
            filename
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        # Load input
        image = np.load(input_path).astype(np.float32)

        # Remove channel dimension if present
        if image.ndim == 3:
            image = image.squeeze()

        if image.ndim != 2:
            print("Skipping invalid file:", filename)
            continue

        # Handle degraded values
        image = np.clip(image, 0.0, 1.0)

        # Tensor: 1 x 1 x H x W
        tensor = torch.from_numpy(image)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(device)

        # Inference
        with torch.no_grad():
            output = model(tensor)

        # Convert to numpy
        output = output.squeeze().cpu().numpy()

        # Ensure valid output
        output = np.nan_to_num(
            output,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        output = np.clip(
            output,
            0.0,
            1.0
        ).astype(np.float32)

        # Save SAME filename as input
        np.save(
            output_path,
            output
        )

        print(
            "Processed:",
            filename,
            "→",
            output.shape
        )

    print("\n✓ All files processed successfully.")


if __name__ == "__main__":
    main()
