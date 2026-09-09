"""Pitch calibration (homography) stage.

kpts.py only exposes a hard-coded __main__ block that walks a whole dataset one
frame at a time.  Almost all of its cost is CPU post-processing (line fitting,
ellipse projection, least-squares refinement) — the GPU forward is ~3% of it —
so this runner shards the frames over several worker processes.

Each shard writes ``<frame>.npy`` next to its own frames, so the shards never
touch the same file.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.getcwd())

# The core keeps weights outside the engine directory, so the default is taken
# from the environment when it is set; the relative path stays as the fallback
# for a plain run inside the engine folder.
CHECKPOINT = os.path.join(os.environ["FG_CHECKPOINT_DIR"],
                          "SoccernetGSR_EfficientNet_Best.pth")     if os.environ.get("FG_CHECKPOINT_DIR") else     "checkpoints/SoccernetGSR_EfficientNet_Best.pth"


def _run_shard(img_dir, result_dir, checkpoint, shard, num_shards, stride):
    """Runs in a worker process: calibrate every num_shards-th frame."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, os.getcwd())
    import kpts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, (os.cpu_count() or 8) // max(1, num_shards)))

    model = kpts.Unet(out_ch=98, num_lines=21)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device).eval()

    transforms = kpts._get_img_transforms(crop_dim=384, is_eval=True)
    dataset = kpts.SoccernetDataset(img_dir, frame_sequence=1, frame_stride=1,
                                    aug_transforms=transforms, hm_size=(384, 384))
    # __getitem__ indexes key_frames and all_image_paths in lockstep (stride 1),
    # so both lists must be sliced the same way
    selected = dataset.all_image_paths[::stride][shard::num_shards]
    dataset.all_image_paths = selected
    dataset.key_frames = selected
    if not selected:
        return 0

    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    kpts.predict_soccernet_inference(
        model, loader, device, result_dir, os.path.basename(img_dir),
        "template/Radar_Dimen.png", "template/soccernet_template_97.npy",
        num_keypoints=97, verbose=False, save_viz=False,
        npy_subfolder="npy_files", viz_subfolder="viz",
    )
    return len(selected)


def _fill_gaps(img_dir, stride):
    """With CALIB_STRIDE > 1, give the skipped frames their neighbour's matrix."""
    import numpy as np

    if stride <= 1:
        return 0
    frames = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    last = None
    filled = 0
    for name in frames:
        stem = os.path.splitext(name)[0]
        npy = os.path.join(img_dir, stem + ".npy")
        if os.path.exists(npy):
            last = npy
        elif last is not None:
            np.save(npy, np.load(last))
            filled += 1
    return filled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", required=True, help=".../data/SoccerNetGS/test")
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--result-dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    for name in sorted(os.listdir(args.split_dir)):
        clip_dir = os.path.join(args.split_dir, name)
        img_dir = os.path.join(clip_dir, "img1")
        if not os.path.isdir(img_dir):
            continue
        result_dir = args.result_dir or os.path.join(clip_dir, "kpts_results")
        os.makedirs(result_dir, exist_ok=True)

        total = len([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
        workers = max(1, min(args.workers, total))
        print(f"[kpts] {name}: {total} frames, {workers} workers, stride {args.stride}",
              flush=True)

        started = time.perf_counter()
        if workers == 1:
            _run_shard(img_dir, result_dir, args.checkpoint, 0, 1, args.stride)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_run_shard, img_dir, result_dir, args.checkpoint,
                                       shard, workers, args.stride)
                           for shard in range(workers)]
                for future in futures:
                    future.result()

        filled = _fill_gaps(img_dir, args.stride)
        done = len([f for f in os.listdir(img_dir) if f.endswith(".npy")])
        print(f"[kpts] {name}: {done}/{total} homographies "
              f"({filled} copied) in {time.perf_counter() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
