import os
import h5py
import hdf5plugin
import numpy as np
import cv2
import yaml
from tqdm import tqdm

def load_timestamps(file_path):
    timestamps = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            start_time, end_time = map(int, line.split(","))
            timestamps.append((start_time, end_time))
    return timestamps

def load_events(event_path):
    with h5py.File(event_path, 'r') as f:
        t_offset = f['t_offset'][()]
        events = {
            't': f['events']['t'][:] + t_offset,
            'x': f['events']['x'][:],
            'y': f['events']['y'][:],
            'p': f['events']['p'][:]
        }
    return events

def create_event_frame(slice_events, frame_shape=(256, 320), downsample=False):
    height, width = frame_shape
    frame = np.ones((height, width, 3), dtype=np.uint8) * 114

    off_events = (slice_events['p'] == -1)
    on_events = (slice_events['p'] == 1)

    frame[slice_events['y'][off_events], slice_events['x'][off_events]] = [0, 0, 255]
    frame[slice_events['y'][on_events], slice_events['x'][on_events]] = [255, 0, 0]

    if downsample:
        frame = cv2.resize(frame, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)

    return frame

def main(config_path):
    # YAMLファイルから設定を読み込み
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    sequence_dir = config['sequence_dir']
    output_dir = config['output_dir']
    frame_shape = tuple(config['frame_shape'])
    downsample = config['downsample']
    num_event_frames = config['num_event_frames']

    os.makedirs(output_dir, exist_ok=True)
    event_path = os.path.join(sequence_dir, "events", "left", "events.h5")
    timestamps_file = os.path.join(sequence_dir, "images", "left", "exposure_timestamps.txt")
    timestamps = load_timestamps(timestamps_file)
    events = load_events(event_path)

    total_frames = len(timestamps)
    event_indices = np.arange(len(events['t']))

    current_event_idx = 0  # イベントの開始インデックス

    for idx in tqdm(range(total_frames), desc="Processing frames"):
        start_time, end_time = timestamps[idx]

        # イベントの範囲を取得
        mask = (events['t'] >= start_time) & (events['t'] < end_time)
        indices = event_indices[mask]

        if len(indices) == 0:
            print(f"Frame {idx} has no events.")
            continue

        # 範囲を分割してイベントフレームを生成
        interval = (end_time - start_time) / num_event_frames
        for i in range(num_event_frames):
            part_start = start_time + i * interval
            part_end = part_start + interval
            part_mask = (events['t'][indices] >= part_start) & (events['t'][indices] < part_end)
            part_indices = indices[part_mask]
            part_events = {k: v[part_indices] for k, v in events.items()}

            event_frame = create_event_frame(part_events, frame_shape, downsample)

            # イベントフレームを保存
            event_frame_filename = os.path.join(
                output_dir, f"event_frame_{idx:06d}_{i}.png")
            cv2.imwrite(event_frame_filename, event_frame)

    print("前処理が完了しました。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="イベントデータの前処理")
    parser.add_argument("--config", type=str, required=True, help="設定ファイルのパス（YAML）")
    args = parser.parse_args()

    main(config_path=args.config)
