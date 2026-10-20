import argparse
from pathlib import Path
import cv2

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def crop_face(frame, margin=0.3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(margin * max(w, h))
    H, W = frame.shape[:2]
    return frame[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=10, help="har N-th frame")
    ap.add_argument("--max_per_video", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in sorted(Path(a.videos).rglob("*.mp4")):
        cap, i, saved = cv2.VideoCapture(str(v)), 0, 0
        name = f"{v.parent.name}_{v.stem}"  # alag method folders me same naam ho sakta hai
        while saved < a.max_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                face = crop_face(frame)
                if face is not None:
                    cv2.imwrite(str(out / f"{name}_frame{i}.png"), face)
                    saved += 1
            i += 1
        cap.release()
    print("done ->", out)


if __name__ == "__main__":
    main()