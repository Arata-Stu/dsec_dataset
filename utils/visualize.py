import numpy as np
import cv2
import random

def save_sequence_as_video_dataloader(dataloader, t_ms: int, output_file: str):
    fps = 1000 / t_ms
    sample_batch = next(iter(dataloader))
    
    # サンプルバッチのサイズを取得し、動画の解像度を決定
    _, h, w = sample_batch['event_frames'][0][0].shape
    size = (w, h)
    
    # 動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, size, isColor=True)
    
    # トラッキングIDに対する色を保存する辞書
    track_colors = {}

    for batch in dataloader:
        event_frames_batch = batch['event_frames']
        labels_batch = batch['labels']
        
        for sample_idx in range(event_frames_batch.shape[0]):
            event_frames = event_frames_batch[sample_idx]
            labels = labels_batch[sample_idx]

            for t, (frame, bbox) in enumerate(zip(event_frames, labels)):
                # フレームの形式を確保
                img_uint = frame.permute(1, 2, 0).numpy().astype('uint8').copy()

                # bboxがNoneでないか、空でないかをチェック
                if bbox is not None and len(bbox) > 0:
                    for box in bbox:
                        try:
                            # 各ボックスの情報を取得
                            x, y, w, h, cls, track_id = box['x'], box['y'], box['w'], box['h'], box['class_id'], box['track_id']
                            if w > 0 and h > 0:
                                # 新しいtrack_idに対してランダムな色を生成
                                if track_id not in track_colors:
                                    track_colors[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                                
                                # 同じtrack_idに対して同じ色を使用
                                color = track_colors[track_id]
                                start_point = (int(x), int(y))
                                end_point = (int(x + w), int(y + h))
                                
                                # 色付き矩形描画
                                cv2.rectangle(img_uint, start_point, end_point, color, 2)
                                cv2.putText(img_uint, f"Cls: {int(cls)} ID: {int(track_id)}", (int(x), int(y) - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                        except Exception as e:
                            print(f"Error drawing rectangle at sample {sample_idx}, time step {t}, box {box}: {e}")
                else:
                    print(f"No valid bounding boxes for sample {sample_idx}, time step {t}")

                # ビデオフレームとして書き込む
                video_writer.write(img_uint)
    
    video_writer.release()
    print(f"動画が保存されました: {output_file}")
