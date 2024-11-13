import os
import cv2
from torch.utils.data import Dataset

class PreprocessedDSECEventImageDataset(Dataset):
    def __init__(self, sequence_dir, event_frame_dir, num_event_frames=5):
        self.sequence_dir = sequence_dir
        self.event_frame_dir = event_frame_dir
        self.image_dir = os.path.join(sequence_dir, "images", "left", "distorted")
        self.num_event_frames = num_event_frames

        # カメラ画像の数を取得
        self.total_frames = len([name for name in os.listdir(self.image_dir) if name.endswith('.png')])

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        # 前処理されたイベントフレームの読み込み
        event_frames = []
        for i in range(self.num_event_frames):
            event_frame_file = os.path.join(
                self.event_frame_dir, f"event_frame_{idx:06d}_{i}.png")
            event_frame = cv2.imread(event_frame_file, cv2.IMREAD_COLOR)
            if event_frame is None:
                raise FileNotFoundError(f"Event frame file {event_frame_file} not found.")
            event_frames.append(event_frame)

        # 通常カメラ画像の読み込み
        image_file = os.path.join(self.image_dir, f"{idx:06d}.png")
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file {image_file} not found.")

        return event_frames, image
