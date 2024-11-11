import torch

def custom_collate_fn(batch):
    # 各サンプルの`event_frames`、`labels`、`file_paths`、`is_start_sequence`、`mask`を個別に取得
    event_frames = [torch.stack([frame.clone().detach() for frame in sample['events']]) for sample in batch]
    labels = [sample['labels'] for sample in batch]
    file_paths = [sample['file_paths'] for sample in batch]
    is_start_sequence = [sample['is_start_sequence'] for sample in batch]
    masks = [sample['mask'].clone().detach().to(dtype=torch.int64) for sample in batch]

    
    # event_framesをバッチ全体でTensorに変換
    event_frames = torch.stack(event_frames)
    
    # masksもバッチ全体でTensorに変換
    masks = torch.stack(masks)

    return {
        'events': event_frames,
        'labels': labels,
        'file_paths': file_paths,
        'is_start_sequence': is_start_sequence,
        'mask': masks
    }
