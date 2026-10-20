import io, random
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TASK_ID = {"ai": 0, "deepfake": 1}


def degrade(img: Image.Image, k: float = 1.0) -> Image.Image:
    """Image ko 'social media' jaisa kharab karo: blur, chhota-bada, JPEG. k=0 band, 1 normal, 2 double."""
    if random.random() < 0.2 * k:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
    if random.random() < 0.2 * k:
        w, h = img.size
        s = random.uniform(0.5, 0.9)
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if random.random() < 0.4 * k:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(30, 100))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


class FakeDataset(Dataset):
    def __init__(self, manifest: str, split: str, train: bool = False, size: int = 224, aug: float = 1.0):
        df = pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.train = train
        self.aug = aug
        # Crop, resize nahi: resize se fake ke nishaan mit jaate hain
        self.crop = T.RandomCrop(size, pad_if_needed=True) if train else T.CenterCrop(size)
        self.norm = T.Compose([T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        img = self.crop(img)
        if self.train:
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
            img = degrade(img, self.aug)
        return self.norm(img), torch.tensor(float(r["label"])), TASK_ID[r["task"]], r["source"]