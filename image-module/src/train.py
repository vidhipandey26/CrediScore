import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset import FakeDataset
from src.model import TruthLensImage
from src.utils import compute_metrics, load_model, predict, save_weights, score, seed_all


def make_loader(manifest, split, train, bs, workers=2, max_n=None, aug=1.0, blur_p=None, blur_max=1.5):
    ds = FakeDataset(manifest, split, train=train, aug=aug, blur_p=blur_p, blur_max=blur_max)
    if len(ds) == 0:
        return None
    if max_n and len(ds) > max_n:
        ds = Subset(ds, torch.randperm(len(ds))[:max_n].tolist())
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=workers, pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blur_p", type=float, default=None, help="blur ki probability alag se (default: 0.2*aug)")
    ap.add_argument("--blur_max", type=float, default=1.5, help="blur ka max radius")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)       # head + forensic branch
    ap.add_argument("--lora_lr", type=float, default=3e-5)  # CLIP ke LoRA adapters
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--clip_name", default="openai/clip-vit-base-patch16")
    ap.add_argument("--no_forensic", action="store_true")
    ap.add_argument("--aug", type=float, default=1.0, help="augmentation: 0 band, 1 normal, 2 double")
    ap.add_argument("--max_train", type=int, default=None, help="quick test ke liye images kam karo")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    seed_all(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    train_dl = make_loader(a.manifest, "train", True, a.bs, a.workers, a.max_train, a.aug, a.blur_p, a.blur_max)
    val_dl = make_loader(a.manifest, "val", False, a.bs, a.workers, a.max_train)
    ood_dl = make_loader(a.manifest, "val_ood", False, a.bs, a.workers, a.max_train)

    model = TruthLensImage(a.clip_name, a.lora_r, not a.no_forensic).to(dev)
    vit_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vit.")]
    oth_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vit.")]
    opt = torch.optim.AdamW([{"params": vit_p, "lr": a.lora_lr}, {"params": oth_p, "lr": a.lr}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lora_lr, a.lr],
                                                total_steps=a.epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    cfg = dict(clip_name=a.clip_name, lora_r=a.lora_r, use_forensic=not a.no_forensic)

    best = -1.0
    for ep in range(a.epochs):
        model.train()
        for x, y, t, _ in tqdm(train_dl, desc=f"epoch {ep + 1}/{a.epochs}"):
            x, y, t = x.to(dev), y.to(dev), t.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                out = model(x)
            logit = out.float().gather(1, t[:, None]).squeeze(1)   # har sample ke apne task ka head
            loss = F.binary_cross_entropy_with_logits(logit, y * 0.9 + 0.05)  # label smoothing 0.1
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        report = {n: compute_metrics(*predict(model, dl, dev)) for n, dl in (("val", val_dl), ("val_ood", ood_dl)) if dl}
        print(json.dumps(report, indent=1))
        sel = score(report.get("val_ood") or report.get("val") or {})  # model selection: unseen generators pe
        if sel > best:
            best = sel
            save_weights(model, a.out, cfg)
            (Path(a.out) / "val_metrics.json").write_text(json.dumps(report, indent=2))
            print(f"saved best (score={sel:.4f})")

    test_dl = make_loader(a.manifest, "test_ood", False, a.bs, a.workers, a.max_train)
    if test_dl:  # final exam: sirf ek baar, best weights ke saath
        res = compute_metrics(*predict(load_model(a.out, dev), test_dl, dev))
        (Path(a.out) / "test_ood_metrics.json").write_text(json.dumps(res, indent=2))
        print("TEST_OOD:", json.dumps(res, indent=1))


if __name__ == "__main__":
    main()