# edit from specforge/data/preprocessing.py
import os

from datasets import Dataset

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from typing import Optional

def list_local_files(path, suffix):
    datapaths = []
    for root, directories, files in os.walk(path):
        part_datapaths = [os.path.join(root, file) for file in files if file.endswith(suffix)]
        datapaths.extend(part_datapaths)
    return datapaths

class OfflineEagleDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, transform=None, max_len=2048):
        self.datapaths = datapath
        self.transform = transform
        self._epoch = 0
        self.max_len = max_len

    def __len__(self):
        return len(self.datapaths)

    def _open_file(self, index):
        return torch.load(self.datapaths[index], weights_only=False)

    def __getitem__(self, index):
        try:
            data = self._open_file(index)
        except Exception as e:
            print(f"ERROR Failed to load {self.datapaths[index]} with error {e}")
            data = self._open_file(0)
            # raise e
        new_data = {}

        # Squeeze due to our data generation script adding a batch dimension
        hidden_states = data["hidden_states"][:self.max_len]

        input_ids = data["input_ids"][1 : self.max_len + 1]
        loss_mask = data["loss_mask"][1 : self.max_len + 1] if "loss_mask" in data else torch.ones_like(input_ids)
        loss_mask[-1] = 0

        pad_size = self.max_len - input_ids.numel()
        pad_zeros = torch.zeros(pad_size, dtype=input_ids.dtype, device=input_ids.device)

        new_data["loss_mask"] = torch.cat([loss_mask, pad_zeros.to(loss_mask.dtype)])
        new_data["attention_mask"] = torch.ones_like(new_data["loss_mask"], dtype=torch.long)
        new_data["hidden_states"] = torch.cat(
            [hidden_states, torch.zeros(self.max_len - hidden_states.shape[0], hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device)],
            dim=0,
        )
        new_data["input_ids"] = torch.cat([input_ids, pad_zeros])
        if self.transform:
            new_data = self.transform(new_data)

        return new_data

    def set_epoch(self, epoch):
        self._epoch = epoch

def build_offline_eagle_dataset(
    hidden_states_path: str,
    max_len: int,
    suffix: str,
) -> torch.utils.data.Dataset:
    return OfflineEagleDataset(
        list_local_files(hidden_states_path, suffix),
        max_len=max_len,
    )


def prepare_dp_dataloaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    pin_memory: Optional[bool] = False,
    shuffle: Optional[bool] = False,
    **dataloader_kwargs
) -> DataLoader:
    """
    Prepare dataloader for distributed data parallel training.

    Args:
        dataset: The dataset to load data from.
        batch_size: The batch size for each GPU.
        num_workers: The number of workers for data loading.
        process_group: The process group for distributed training.
        pin_memory: Whether to pin memory for data loading.
        shuffle: Whether to shuffle the dataset.
        is_vlm: Whether the dataset is a vision-language model dataset.
        **dataloader_kwargs: Additional keyword arguments for the DataLoader.

    Returns:
        A DataLoader for the dataset.
    """
    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs
    )
    return dataloader
