import torch
from torch.utils.data import DataLoader

def dataset_to_dataloader(dataset, samples_per_class = 5, tot_samples_per_class=50, batch_size=8, shuffle=False, num_workers=8):
    # Take only a subset (here 5000 samples, 1 per class)
    if samples_per_class is None:
        dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    else:
        step = tot_samples_per_class//samples_per_class
        index = list(range(0, len(dataset), step))
        dataset = torch.utils.data.Subset(dataset, index)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    return dataloader