import io, random
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TASK_ID = {"ai": 0, "deepfake": 1}


def jpeg(img: Image.Image, q: int) -> Image.Image:
    """Image ko JPEG-machine se guzaaro (quality q)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade(img: Image.Image, k: float = 1.0, blur_p=None, blur_max: float = 1.5) -> Image.Image:
    """Image ko 'social media' jaisa kharab karo: blur, chhota-bada, JPEG. k=0 band, 1 normal, 2 double."""
    p_blur = 0.2 * k if blur_p is None else blur_p   # blur_p diya ho to sirf blur ki probability alag
    if random.random() < p_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, blur_max)))
    if random.random() < 0.2 * k:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
    if random.random() < 0.2 * k:
        w, h = img.size
        s = random.uniform(0.5, 0.9)
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if random.random() < 0.4 * k:
        img = jpeg(img, random.randint(30, 100))
    return img


class FakeDataset(Dataset):
    def __init__(self, manifest: str, split: str, train: bool = False, size: int = 224,
                 aug: float = 1.0, jpeg_all: bool = True, blur_p=None, blur_max: float = 1.5):
        df = pd.read_csv(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.train = train
        self.aug = aug
        self.size = size
        self.jpeg_all = jpeg_all
        self.blur_p = blur_p
        self.blur_max = blur_max
        # Crop, resize nahi: resize se fake ke nishaan mit jaate hain
        self.crop = T.RandomCrop(size, pad_if_needed=True) if train else T.CenterCrop(size)
        self.norm = T.Compose([T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        if min(img.size) < self.size:  # chhoti image: black padding ki jagah upscale (real/fake dono ke liye same rule)
            s = self.size / min(img.size)
            img = img.resize((max(self.size, round(img.width * s)), max(self.size, round(img.height * s))), Image.BICUBIC)
        if self.jpeg_all:  # real (JPEG) aur fake (PNG) dono ko ek hi JPEG machine se guzaaro -> file-type ka clue khatam
            img = jpeg(img, random.randint(70, 100) if self.train else 90)
        img = self.crop(img)
        if self.train:
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
            img = degrade(img, self.aug, self.blur_p, self.blur_max)
        return self.norm(img), torch.tensor(float(r["label"])), TASK_ID[r["task"]], r["source"]