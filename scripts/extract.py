import sys, time, numpy as np, cv2, mediapipe as mp
from pathlib import Path
sys.path.insert(0, ".")
from asl_benchmark.loader import load

TRIM = False
out_dir = Path("data/wlasl100/keypoints"); out_dir.mkdir(parents=True, exist_ok=True)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                 min_detection_confidence=0.5)
todo = [s for s in load("wlasl100", "data", download=False).samples
        if s.video_path.exists() and not (out_dir / f"{s.sample_id}.npy").exists()]
print(len(todo), "videos to process")
t0 = time.time()
for k, s in enumerate(todo, 1):
    cap, frames, i = cv2.VideoCapture(str(s.video_path)), [], 0
    while True:
        ok, img = cap.read()
        if not ok: break
        i += 1
        if TRIM and (i < s.frame_start or (s.frame_end != -1 and i > s.frame_end)):
            continue
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        row = np.full((42, 3), np.nan, np.float32)        # 0-20 left hand, 21-41 right hand
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                # MediaPipe's "Left" is the signer's right hand on normal video
                start = 21 if hd.classification[0].label == "Left" else 0
                row[start:start + 21] = [[p.x, p.y, p.z] for p in lm.landmark]
        frames.append(row)
    cap.release()
    np.save(out_dir / f"{s.sample_id}.npy", np.array(frames, np.float32).reshape(-1, 42, 3))
    seen = np.isfinite(np.array(frames)[..., 0]).any(-1).mean() if frames else 0
    print(f"[{k}/{len(todo)}] {s.sample_id} frames={len(frames)} hand_seen={seen:.0%} "
          f"elapsed={time.time()-t0:.0f}s")
