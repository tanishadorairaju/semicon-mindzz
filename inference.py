import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn


# ==========================================
# NAFNET ARCHITECTURE
# ==========================================

class LayerNorm2d(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(1, channels, 1, 1)
        )

        self.bias = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    def forward(self, x):

        mean = x.mean(
            dim=(1, 2, 3),
            keepdim=True
        )

        var = (
            x - mean
        ).pow(2).mean(
            dim=(1, 2, 3),
            keepdim=True
        )

        x = (x - mean) / torch.sqrt(
            var + 1e-6
        )

        return x * self.weight + self.bias


class SimpleGate(nn.Module):

    def forward(self, x):

        x1, x2 = x.chunk(2, dim=1)

        return x1 * x2


class NAFBlock(nn.Module):

    def __init__(
        self,
        c,
        dw_expand=2,
        ffn_expand=2
    ):

        super().__init__()

        dw_channel = c * dw_expand

        self.norm1 = LayerNorm2d(c)

        self.conv1 = nn.Conv2d(
            c, dw_channel, 1, 1, 0
        )

        self.conv2 = nn.Conv2d(
            dw_channel,
            dw_channel,
            3,
            1,
            1,
            groups=dw_channel
        )

        self.sg = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                dw_channel // 2,
                dw_channel // 2,
                1,
                1,
                0
            )
        )

        self.conv3 = nn.Conv2d(
            dw_channel // 2,
            c,
            1,
            1,
            0
        )

        self.norm2 = LayerNorm2d(c)

        ffn_channel = ffn_expand * c

        self.conv4 = nn.Conv2d(
            c,
            ffn_channel,
            1,
            1,
            0
        )

        self.conv5 = nn.Conv2d(
            ffn_channel // 2,
            c,
            1,
            1,
            0
        )

        self.beta = nn.Parameter(
            torch.zeros(1, c, 1, 1)
        )

        self.gamma = nn.Parameter(
            torch.zeros(1, c, 1, 1)
        )

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
            img_channel,
            width,
            3,
            1,
            1
        )

        self.ending = nn.Conv2d(
            width,
            img_channel,
            3,
            1,
            1
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width

        for num in enc_blk_nums:

            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(chan)
                        for _ in range(num)
                    ]
                )
            )

            self.downs.append(
                nn.Conv2d(
                    chan,
                    chan * 2,
                    2,
                    2
                )
            )

            chan *= 2

        self.middle = nn.Sequential(
            *[
                NAFBlock(chan)
                for _ in range(middle_blk_num)
            ]
        )

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num in dec_blk_nums:

            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        chan,
                        chan * 2,
                        1,
                        1,
                        0
                    ),
                    nn.PixelShuffle(2)
                )
            )

            chan //= 2

            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(chan)
                        for _ in range(num)
                    ]
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
            nn.Conv2d(
                1,
                4,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2)
        )

    def forward(self, x):

        x = self.nafnet(x)
        x = self.upscale(x)

        return x


# ==========================================
# LOAD MODEL
# ==========================================

def load_model(model_path, device):

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

    return model


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def load_image(path):

    image = Image.open(path).convert("L")

    image = np.array(
        image
    ).astype(np.float32) / 255.0

    image = np.clip(
        image,
        0.0,
        1.0
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    return tensor


# ==========================================
# MAIN INFERENCE
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--model",
        default="model/nafnet_final.pth"
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = load_model(
        args.model,
        device
    )

    supported = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    )

    image_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith(supported)
    ]

    print(
        "Images found:",
        len(image_files)
    )

    for filename in image_files:

        input_path = os.path.join(
            args.input_dir,
            filename
        )

        output_name = (
            os.path.splitext(filename)[0]
            + "_restored.png"
        )

        output_path = os.path.join(
            args.output_dir,
            output_name
        )

        image = load_image(
            input_path
        ).to(device)

        with torch.no_grad():

            restored = model(image)

        restored = restored.squeeze().cpu().numpy()

        restored = np.clip(
            restored,
            0.0,
            1.0
        )

        restored = (
            restored * 255
        ).astype(np.uint8)

        Image.fromarray(
            restored
        ).save(output_path)

        print(
            "Restored:",
            filename,
            "→",
            output_name
        )

    print("\n✓ Inference completed.")


if __name__ == "__main__":
    main()
