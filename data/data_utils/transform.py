import torch
import numpy as np
import cv2


class EventPadderTransform:
    def __init__(self, desired_size, mode='constant', value=0):
        """
        :param desired_size: Desired size (height=width) for square padding.
        :param mode: Padding mode (e.g., 'constant', 'reflect', 'replicate').
        :param value: Padding value when mode is 'constant'.
        """
        assert isinstance(desired_size, int), "desired_size should be an integer representing the side length of the square"
        self.desired_size = desired_size
        self.mode = mode
        self.value = value

    def _pad_numpy(self, input_array):
        """
        Pads the input array to the desired square size using numpy.
        :param input_array: NumPy array of shape (channels, height, width).
        :return: Padded NumPy array of shape (channels, desired_size, desired_size).
        """
        ht, wd = input_array.shape[-2:]  # Get current height and width.
        
        # Ensure the desired dimension is greater than or equal to the current dimensions.
        assert ht <= self.desired_size, f"Current height {ht} exceeds desired size {self.desired_size}"
        assert wd <= self.desired_size, f"Current width {wd} exceeds desired size {self.desired_size}"

        # Calculate padding amounts, padding on right and bottom to make it square
        pad_top = 0
        pad_left = 0
        pad_bottom = self.desired_size - ht
        pad_right = self.desired_size - wd

        # Define padding for each dimension: (channels, height, width)
        pad_width = [(0, 0),           # No padding for channels
                     (pad_top, pad_bottom),  # Padding for height
                     (pad_left, pad_right)]  # Padding for width

        # Apply padding
        padded_array = np.pad(input_array, pad_width, mode=self.mode, constant_values=self.value)
        
        return padded_array

    def __call__(self, sample):
        """
        Apply square padding to each event frame in 'event_frames' and keep 'labels' unchanged.
        :param sample: Dictionary containing 'event_frames' and 'labels'.
        :return: Dictionary with each 'event_frame' padded to a square size.
        """
        event_frames = sample['events']
        labels_sequence = sample['labels']

        # Ensure the event frames are tensors, so convert to numpy for padding
        padded_event_frames = []
        for frame in event_frames:
            # Convert to numpy, pad, and convert back to tensor
            frame_np = frame.numpy()
            padded_frame_np = self._pad_numpy(frame_np)
            padded_frame = torch.from_numpy(padded_frame_np)
            padded_event_frames.append(padded_frame)

        # Stack padded frames to create a tensor with consistent shape
        padded_event_frames = torch.stack(padded_event_frames)

        # Update sample with padded frames
        sample['event'] = padded_event_frames
        sample['labels'] = labels_sequence  # Labels remain unchanged
        return sample


def flip_horizontal(image, labels):
    """
    画像とラベルを水平方向に反転します。

    Args:
        image (numpy.ndarray): 形状が (D, H, W) の画像データ。
        labels (numpy.ndarray or None): 各要素が1つの辞書を含むnumpy.ndarray形式のラベル。

    Returns:
        flipped_image (numpy.ndarray): 反転後の画像。
        flipped_labels (numpy.ndarray or None): 調整されたラベルの配列、または None。
    """

    D, H, W = image.shape

    # 各チャネルを反転
    flipped_channels = []
    for d in range(D):
        channel = image[d]
        flipped_channel = cv2.flip(channel, 1)  # 水平方向に反転
        flipped_channels.append(flipped_channel)
    flipped_image = np.stack(flipped_channels, axis=0)

    if labels is not None:
        # `labels`がnumpy.ndarrayで各要素が辞書である前提で処理
        flipped_labels = labels.copy()
        for label in flipped_labels:
            # ラベルのコピーを作成し、パディング値（x=0, w=0）のラベルはそのまま
            if label['x'] != 0 or label['w'] != 0:
                label['x'] = W - label['x'] - label['w']
        return flipped_image, flipped_labels
    else:
        return flipped_image, None


def rotate(image, labels, angle):
    """
    画像とラベルを指定した角度だけ回転します。

    Args:
        image (numpy.ndarray): 形状が (D, H, W) の画像データ。
        labels (numpy.ndarray or None): 各要素が辞書形式のnumpy.ndarrayラベルデータ。
        angle (float): 回転角度（度単位）。正の値は反時計回りの回転を意味します。

    Returns:
        rotated_image (numpy.ndarray): 回転後の画像。
        rotated_labels (numpy.ndarray or None): 回転後のラベル、または None。
    """
    D, H, W = image.shape
    center = (W / 2, H / 2)

    # 回転行列を作成
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # 各チャネルを個別に回転
    rotated_channels = [cv2.warpAffine(image[d], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for d in range(D)]
    rotated_image = np.stack(rotated_channels, axis=0)

    if labels is not None:
        rotated_labels = labels.copy()
        new_boxes = []

        # 回転行列を計算
        theta = np.deg2rad(-angle)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        for label in rotated_labels:
            x, y, w, h = label['x'], label['y'], label['w'], label['h']
            corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

            # 中心に合わせてコーナーをシフトして回転
            corners_centered = corners - center
            rotated_corners = np.dot(corners_centered, rotation_matrix.T) + center

            # 回転後のバウンディングボックスを計算
            x_min, y_min = rotated_corners[:, 0].min(), rotated_corners[:, 1].min()
            x_max, y_max = rotated_corners[:, 0].max(), rotated_corners[:, 1].max()

            new_boxes.append({'x': x_min, 'y': y_min, 'w': x_max - x_min, 'h': y_max - y_min})

        # 回転後のバウンディングボックスで`labels`を更新
        for label, new_box in zip(rotated_labels, new_boxes):
            label.update(new_box)

        return rotated_image, rotated_labels
    else:
        return rotated_image, None



def clip_bboxes(labels, image_shape):
    """
    Clip bounding boxes to ensure they remain within the image bounds.

    Args:
        labels (numpy.ndarray or None): 各要素が辞書形式のラベル配列。
        image_shape (Tuple[int, int]): Shape of the image (height, width).

    Returns:
        clipped_labels (numpy.ndarray or None): 調整されたバウンディングボックス、または None。
    """
    if labels is None:
        return None

    clipped_labels = labels.copy()
    height, width = image_shape

    for label in clipped_labels:
        # x, y 座標をクリップし、画像の範囲内に調整
        label['x'] = np.clip(label['x'], 0, width)
        label['y'] = np.clip(label['y'], 0, height)

        # w, h も画像の範囲内に調整
        label['w'] = np.clip(label['w'], 0, width - label['x'])
        label['h'] = np.clip(label['h'], 0, height - label['y'])

    return clipped_labels


def remove_flat_labels(labels):
    """
    Remove flat labels (where w <= 0 or h <= 0) from the labels array.

    Args:
        labels (numpy.ndarray or None): 各要素が辞書形式のラベル配列。

    Returns:
        filtered_labels (numpy.ndarray or None): w > 0 かつ h > 0 のラベルのみを含む配列。
    """
    if labels is None:
        return None

    # w > 0 かつ h > 0 の条件を満たすラベルのみを残す
    filtered_labels = [label for label in labels if label['w'] > 0 and label['h'] > 0]

    # ラベルが存在しない場合は None を返す
    return np.array(filtered_labels) if filtered_labels else None


def find_zoom_center(labels_list):
    """
    全フレームのラベルからズームの中心を計算します。

    Args:
        labels_list (list): 各フレームに対するラベルリスト。

    Returns:
        Tuple[int, int] or None: ズームの中心座標 (x, y)、または有効なラベルがない場合は None。
    """
    possible_centers = []

    for labels in labels_list:
        if labels is not None and len(labels) > 0:
            for label in labels:
                center_x = label['x'] + label['w'] / 2
                center_y = label['y'] + label['h'] / 2
                possible_centers.append((center_x, center_y))

    if possible_centers:
        avg_center_x = int(np.mean([c[0] for c in possible_centers]))
        avg_center_y = int(np.mean([c[1] for c in possible_centers]))
        return avg_center_x, avg_center_y

    return None


def zoom_in(image, labels, zoom_factor, center=None):
    """
    画像をズームインし、ラベルを調整します。

    Args:
        image (numpy.ndarray): 形状が (D, H, W) の画像データ。
        labels (numpy.ndarray or None): 各要素が辞書形式のラベルデータ。
        zoom_factor (float): ズームイン倍率（>1）。
        center (Tuple[int, int], optional): ズームの中心座標 (x, y)。

    Returns:
        zoomed_image (numpy.ndarray): ズームイン後の画像。
        zoomed_labels (numpy.ndarray or None): 調整されたラベル、または None。
    """
    D, H, W = image.shape
    new_H, new_W = int(H / zoom_factor), int(W / zoom_factor)

    # ズームウィンドウの位置を計算
    if center is None:
        x1 = np.random.randint(0, W - new_W + 1)
        y1 = np.random.randint(0, H - new_H + 1)
    else:
        cx, cy = center
        x1 = max(0, min(int(cx - new_W // 2), W - new_W))
        y1 = max(0, min(int(cy - new_H // 2), H - new_H))

    # 各チャネルをクロップしてリサイズ
    zoomed_channels = [cv2.resize(image[d, y1:y1 + new_H, x1:x1 + new_W], (W, H), interpolation=cv2.INTER_CUBIC) for d in range(D)]
    zoomed_image = np.stack(zoomed_channels, axis=0)

    if labels is not None:
        zoomed_labels = labels.copy()
        for label in zoomed_labels:
            label['x'] = (label['x'] - x1) * zoom_factor
            label['y'] = (label['y'] - y1) * zoom_factor
            label['w'] *= zoom_factor
            label['h'] *= zoom_factor

        # バウンディングボックスをクリップ
        zoomed_labels = clip_bboxes(zoomed_labels, (H, W))
        zoomed_labels = remove_flat_labels(zoomed_labels)
    else:
        zoomed_labels = None

    return zoomed_image, zoomed_labels

def zoom_out(image, labels, zoom_factor, center=None):
    """
    画像をズームアウトし、ラベルを調整します。

    Args:
        image (numpy.ndarray): 形状が (D, H, W) の画像データ。
        labels (numpy.ndarray or None): 各要素が辞書形式のラベルデータ。
        zoom_factor (float): ズームアウト倍率（>1）。
        center (Tuple[int, int], optional): ズームの中心座標 (x, y)。

    Returns:
        zoomed_image (numpy.ndarray): ズームアウト後の画像。
        zoomed_labels (numpy.ndarray or None): 調整されたラベル、または None。
    """
    D, H, W = image.shape
    new_H, new_W = int(H / zoom_factor), int(W / zoom_factor)

    # 各チャネルをリサイズ
    resized_channels = [cv2.resize(image[d], (new_W, new_H), interpolation=cv2.INTER_CUBIC) for d in range(D)]
    resized_image = np.stack(resized_channels, axis=0)

    # キャンバスを作成してズームアウト画像を配置
    canvas = np.zeros_like(image)
    if center is None:
        x1 = np.random.randint(0, W - new_W + 1)
        y1 = np.random.randint(0, H - new_H + 1)
    else:
        cx, cy = center
        x1 = max(0, min(int(cx - new_W // 2), W - new_W))
        y1 = max(0, min(int(cy - new_H // 2), H - new_H))

    canvas[:, y1:y1 + new_H, x1:x1 + new_W] = resized_image

    if labels is not None:
        zoomed_labels = labels.copy()
        for label in zoomed_labels:
            label['x'] = label['x'] / zoom_factor + x1
            label['y'] = label['y'] / zoom_factor + y1
            label['w'] /= zoom_factor
            label['h'] /= zoom_factor

        # バウンディングボックスをクリップ
        zoomed_labels = clip_bboxes(zoomed_labels, (H, W))
        zoomed_labels = remove_flat_labels(zoomed_labels)
    else:
        zoomed_labels = None

    return canvas, zoomed_labels


class RandomSpatialAugmentor:
    def __init__(self,
                 h_flip_prob=0.5,
                 rotation_prob=0.5,
                 rotation_angle_range=(-6, 6),
                 zoom_in_weight=8,
                 zoom_out_weight=2,
                 zoom_in_range=(1.0, 1.5),
                 zoom_out_range=(1.0, 1.5),
                 zoom_prob=0.0):
        self.h_flip_prob = h_flip_prob
        self.rotation_prob = rotation_prob
        self.rotation_angle_range = rotation_angle_range
        self.zoom_in_weight = zoom_in_weight
        self.zoom_out_weight = zoom_out_weight
        self.zoom_in_range = zoom_in_range
        self.zoom_out_range = zoom_out_range
        self.zoom_prob = zoom_prob

        # Zoom operation weighted distribution
        self.zoom_in_or_out_distribution = torch.distributions.categorical.Categorical(
            probs=torch.tensor([zoom_in_weight, zoom_out_weight], dtype=torch.float)
        )

    def __call__(self, sample):
        event_frames = sample['events']  # シーケンス (sequence_length, channels, height, width)
        labels_sequence = sample['labels']     # シーケンス内のラベルリスト

        augmented_event_frames = []
        augmented_labels_sequence = []

        # Decide random transformation parameters once
        apply_h_flip = np.random.rand() < self.h_flip_prob
        apply_rotation = np.random.rand() < self.rotation_prob
        angle = np.random.uniform(*self.rotation_angle_range) if apply_rotation else None
        apply_zoom = np.random.rand() < self.zoom_prob

        # Zoom settings
        zoom_func = None
        zoom_factor = None
        if apply_zoom:
            zoom_choice = self.zoom_in_or_out_distribution.sample().item()
            if zoom_choice == 0:  # Zoom In
                zoom_factor = np.random.uniform(*self.zoom_in_range)
                zoom_func = zoom_in
            else:  # Zoom Out
                zoom_factor = np.random.uniform(*self.zoom_out_range)
                zoom_func = zoom_out

        # Determine zoom center coordinates (common for all frames)
        zoom_center = find_zoom_center(labels_sequence)
        _, H, W = event_frames[0].shape
        if zoom_center is None:
            zoom_center = (W // 2, H // 2)

        # Apply the same preprocessing to each frame
        for frame, labels in zip(event_frames, labels_sequence):
            frame_np = frame.numpy()

            # Horizontal Flip
            if apply_h_flip:
                frame_np, labels = flip_horizontal(frame_np, labels)

            # Rotation
            if apply_rotation:
                frame_np, labels = rotate(frame_np, labels, angle)

            # Zoom In/Out
            if zoom_func is not None:
                frame_np, labels = zoom_func(frame_np, labels, zoom_factor, center=zoom_center)

            processsed_labels = clip_bboxes(labels, (H, W))
            processsed_labels = remove_flat_labels(processsed_labels)
            # Convert back to tensor and add to the list
            augmented_event_frames.append(torch.from_numpy(frame_np))
            augmented_labels_sequence.append(processsed_labels)

        # Stack augmented frames to create a tensor with consistent shape
        augmented_event_frames = torch.stack(augmented_event_frames)

        # Update sample with augmented frames and labels
        sample['events'] = augmented_event_frames
        sample['labels'] = augmented_labels_sequence

        return sample
