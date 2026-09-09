# This script produces tracklets given tracking results and original sequence frame as RGB images.
import argparse

import os
from tqdm import tqdm
from loguru import logger
from PIL import Image

import pickle
import numpy as np
import glob

import torch
import torchvision.transforms as T

from Tracklet import Tracklet

try:
    from fast_prep import shared_reid, thread_map
except ImportError:      # running outside the engine package
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fast_prep import shared_reid, thread_map


# Frames are decoded and their crops prepared a block at a time, so the pool has
# real work to spread; the GPU then gets one full batch instead of one call per
# frame, which on this footage was twenty crops at a time.
FRAME_BLOCK = 32
REID_BATCH = 256


def generate_tracklets(cfg, datasets):
    idatr_cfg = cfg['IDATR']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = shared_reid('osnet_x1_0', idatr_cfg['MODEL_PATH'], device)
    val_transforms = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for split in datasets:
        data_path = os.path.join(cfg['DATA_DIR'], split)
        seqs = sorted(glob.glob(os.path.join(data_path, '**/*rmved_SNGS*.txt'), recursive=True))
        for s_id, seq in tqdm(enumerate(seqs, 1), total=len(seqs), desc='Processing Seqs'):
            seq_name = seq.split("_")[-1].replace('.txt', '')
            seq_dir = os.path.dirname(seq)
            imgs = sorted(glob.glob(os.path.join(seq_dir, 'img1', '*.jpg')))
            track_res = np.genfromtxt(seq, dtype=str, delimiter=',', encoding="utf-8")
            last_frame = int(track_res[-1][0])
            seq_tracks = {}
            frame_column = track_res[:, 0].astype(int)   # parsed once, not per frame

            # One feature per track per frame, appended in frame order -- the
            # order the detections went in, which is what append_feat pairs with.
            pending_tensors, pending_tracks = [], []

            def flush():
                if not pending_tensors:
                    return
                features = extractor(torch.stack(pending_tensors))
                feats = features.cpu().detach().numpy()
                for track_id, feat in zip(pending_tracks, feats):
                    feat = feat / np.linalg.norm(feat)
                    seq_tracks[track_id].append_feat(feat)
                pending_tensors.clear()
                pending_tracks.clear()

            for block_start in range(1, last_frame + 1, FRAME_BLOCK):
                block = list(range(block_start, min(block_start + FRAME_BLOCK, last_frame + 1)))
                # decoding a frame is its slowest part and holds no lock
                images = thread_map(
                    lambda fid: Image.open(imgs[int(fid) - 1]).convert('RGB'), block)

                block_crops, block_tracks = [], []
                for frame_id, img in zip(block, images):
                    if frame_id % 100 == 0:
                        logger.info(f'Processing frame {frame_id}/{last_frame}')
                    inds = frame_column == frame_id
                    frame_res = track_res[inds]
                    if not len(frame_res):
                        print(f"No detection at frame: {frame_id}")
                        continue

                    frame_crops = {}
                    for idx, (frame, track_id, l, t, w, h, score, role, jn, jc, team) in enumerate(frame_res):  # jn: jersey number, jc: jersey color
                        # Update tracklet with detection
                        frame, track_id, l, t, w, h, score = \
                            int(frame), int(track_id), float(l), float(t), float(w), float(h), float(score)
                        bbox = [l, t, w, h]
                        if track_id not in seq_tracks:
                            seq_tracks[track_id] = Tracklet(track_id, frame, score, bbox, role, jn, jc, team)
                        else:
                            seq_tracks[track_id].append_det(frame, score, bbox, role, jn, jc, team)
                        # a track seen twice in one frame keeps the last crop,
                        # exactly as it did when the batch was indexed by track
                        frame_crops[track_id] = img.crop((l, t, l + w, t + h))

                    block_crops.extend(frame_crops.values())
                    block_tracks.extend(frame_crops.keys())

                if not block_crops:
                    continue
                # resize and normalise: PIL and numpy work, so the pool helps
                prepared = thread_map(val_transforms, block_crops)
                for tensor, track_id in zip(prepared, block_tracks):
                    pending_tensors.append(tensor)
                    pending_tracks.append(track_id)
                    if len(pending_tensors) >= REID_BATCH:
                        flush()
            flush()

            # save seq_tracks into pickle file
            track_output_path = os.path.join(seq_dir, f'{seq_name}.pkl')
            with open(track_output_path, 'wb') as f:
                pickle.dump(seq_tracks, f)
            logger.info(f"save tracklets info to {track_output_path}")


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    data_sets = cfg['DATA_SETS']
    generate_tracklets(cfg, data_sets)
