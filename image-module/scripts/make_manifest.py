"""Example:
python scripts/make_manifest.py \
  --add ai:genimage_sd14:/path/real:/path/fake \
  --add deepfake:ffpp:/path/real_faces:/path/fake_faces \
  --val_ood genimage_biggan --test_ood celebdf --max_per_class 5000
"""
import argparse, hashlib, random, re
from pathlib import Path
import pandas as pd

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def group_id(p: Path) -> str:
    # video frames ("<video>_frame12.png") ek hi group me jaate hain -> same split me rahenge
    return str(p.parent) + "/" + re.sub(r"_(frame|f)\d+$", "", p.stem)


def bucket(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 100


def list_images(folder: str, limit: int, seed: int):
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXT)
    if limit and len(files) > limit:
        files = random.Random(seed).sample(files, limit)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", required=True, help="task:source:real_dir:fake_dir")
    ap.add_argument("--val_ood", nargs="*", default=[], help="sources jo sirf model-selection ke liye hain")
    ap.add_argument("--test_ood", nargs="*", default=[], help="sources jo sirf final test ke liye hain")
    ap.add_argument("--max_per_class", type=int, default=0)
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = []
    for spec in a.add:
        task, source, real_dir, fake_dir = spec.split(":", 3)
        assert task in ("ai", "deepfake"), "task 'ai' ya 'deepfake' hona chahiye"
        for label, folder in ((0, real_dir), (1, fake_dir)):
            for p in list_images(folder, a.max_per_class, a.seed):
                if source in a.test_ood:
                    split = "test_ood"
                elif source in a.val_ood:
                    split = "val_ood"
                else:
                    b = bucket(group_id(p))
                    split = "train" if b < 80 else "val" if b < 90 else "test_in"
                rows.append(dict(path=str(p), label=label, task=task, source=source, split=split))
    assert rows, "Koi image nahi mili - --add me diye folder paths check karo"
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby(["split", "task", "label"]).size())
    print("saved ->", a.out)


if __name__ == "__main__":
    main()