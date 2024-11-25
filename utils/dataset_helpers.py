import torch
import random
from torch.utils.data import DataLoader
def dataset_to_dataloader(dataset, samples_per_class = 5, tot_samples_per_class=50, batch_size=8, shuffle=False, num_workers=8, seed=42):    
    if samples_per_class is None:
        dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    else:
        # Take a subset
        random.seed(seed)
        all_indices = list(range(len(dataset)))
        nr_classes = len(dataset.classes)
        index = [random.sample(all_indices[x*tot_samples_per_class:(x + 1)*tot_samples_per_class], samples_per_class) for x in range(nr_classes)]
        index = [x for xs in index for x in xs]
        dataset = torch.utils.data.Subset(dataset, index)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    return dataloader