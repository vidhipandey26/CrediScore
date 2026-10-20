import json, random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

TASKS = ["ai", "deepfake"]


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys, ts, srcs = [], [], [], []
    for x, y, t, src in tqdm(loader, leave=False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model(x.to(device))
        logit = out.float().gather(1, t.to(device)[:, None]).squeeze(1)
        probs.append(torch.sigmoid(logit).cpu()); ys.append(y); ts.append(t); srcs += list(src)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy(), torch.cat(ts).numpy(), np.array(srcs)


def _safe(fn, y, p):
    try:
        return float(fn(y, p))
    except ValueError:  # sirf ek class ho to AUC/AP nahi nikalta
        return float("nan")


def compute_metrics(p, y, t, s):
    res = {}
    for tid, name in enumerate(TASKS):
        m = t == tid
        if m.sum() == 0:
            continue
        pp, yy, ss = p[m], y[m], s[m]
        res[name] = {"n": int(m.sum()), "acc": float(((pp > 0.5) == yy).mean()),
                     "auc": _safe(roc_auc_score, yy, pp), "ap": _safe(average_precision_score, yy, pp),
                     "per_source_acc": {str(k): float(((pp[ss == k] > 0.5) == yy[ss == k]).mean()) for k in np.unique(ss)}}
    return res


def score(res):
    vals = [v["ap"] for v in res.values() if v["ap"] == v["ap"]]  # nan hatao
    return float(np.mean(vals)) if vals else 0.0


def save_weights(model, out_dir, cfg):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    keep = {k: v.cpu() for k, v in model.state_dict().items()
            if "lora_" in k or k.startswith(("forensic", "fuse"))}  # sirf chhota hissa save: LoRA + heads
    torch.save(keep, out / "weights.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))


def load_model(weights_dir, device="cpu"):
    from src.model import TruthLensImage
    d = Path(weights_dir)
    cfg = json.loads((d / "config.json").read_text())
    model = TruthLensImage(cfg["clip_name"], cfg["lora_r"], cfg["use_forensic"])
    model.load_state_dict(torch.load(d / "weights.pt", map_location="cpu"), strict=False)
    return model.to(device).eval()