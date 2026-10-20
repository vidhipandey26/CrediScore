import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPVisionModel


class ForensicBranch(nn.Module):
    """Chhota CNN jo image ke 'noise' (high-pass residual) ko dekhta hai."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        k = torch.tensor([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                          [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
        self.register_buffer("hp", k.view(1, 1, 5, 5).repeat(3, 1, 1, 1))

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.net = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, out_dim),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(F.conv2d(x, self.hp, padding=2, groups=3))


class TruthLensImage(nn.Module):
    """Output [B, 2]: column 0 = AI-generated head, column 1 = deepfake head."""

    def __init__(self, clip_name="openai/clip-vit-base-patch16", lora_r=8, use_forensic=True):
        super().__init__()
        vit = CLIPVisionModel.from_pretrained(clip_name)
        d = vit.config.hidden_size
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
        self.vit = get_peft_model(vit, cfg)  # base freeze, sirf LoRA train hoga
        self.use_forensic = use_forensic
        self.forensic = ForensicBranch(256) if use_forensic else None
        self.fuse = nn.Sequential(nn.Linear(d + (256 if use_forensic else 0), 512), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(512, 2))

    def forward(self, x):
        f = self.vit(pixel_values=x).pooler_output
        if self.use_forensic:
            f = torch.cat([f, self.forensic(x)], dim=1)
        return self.fuse(f)