import argparse, json
import torch
from src.train import make_loader
from src.utils import compute_metrics, load_model, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test_ood")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl = make_loader(a.manifest, a.split, False, a.bs)
    assert dl is not None, f"split '{a.split}' me koi image nahi mili"
    print(json.dumps(compute_metrics(*predict(load_model(a.weights, dev), dl, dev)), indent=2))


if __name__ == "__main__":
    main()