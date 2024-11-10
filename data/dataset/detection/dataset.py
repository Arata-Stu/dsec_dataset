import os
import yaml
from torch.utils.data import ConcatDataset
from .sequence import SequenceDataset

import os
import yaml
from torch.utils.data import ConcatDataset

class DsecConcatDataset(ConcatDataset):
    def __init__(self, config_path: str, base_data_dir: str, mode: str, sequence_length: int = 1, guarantee_label: bool = False, transform=None):
        """
        Args:
            config_path (str): YAMLファイルのパス。
            base_data_dir (str): ベースのデータディレクトリのパス。
            mode (str): 'train', 'val', 'test' のいずれか。
            sequence_length (int): シーケンスの長さ。
            guarantee_label (bool): True の場合、ラベルが存在するシーケンスのみを含める。
        """
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.base_data_dir = base_data_dir
        self.sequence_length = sequence_length
        self.guarantee_label = guarantee_label
        self.mode = mode

        # 指定されたモードのシーケンスディレクトリリストを取得
        split_sequences = config['splits'].get(mode, [])
        
        # 各シーケンスディレクトリに対してSequenceDatasetを作成し、リストに追加
        datasets = [
            SequenceDataset(
                data_dir=os.path.join(self.base_data_dir, sequence),
                mode=self.mode,
                sequence_length=self.sequence_length,
                guarantee_label=self.guarantee_label,
                transform=transform,
            )
            for sequence in split_sequences
        ]
        
        # ConcatDatasetの初期化を利用して複数のデータセットを結合
        super().__init__(datasets)
