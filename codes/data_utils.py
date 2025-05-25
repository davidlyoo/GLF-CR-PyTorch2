import torch
from torch.utils.data import DataLoader, Subset

def load_data(
    opts,
    frac,
    seed,
    data_dir,
    dataset,
    batch_size,
    include_test=False,
    num_workers=16,
):
    root_dir = data_dir

    shuffle_gen = torch.Generator().manual_seed(seed)

    from sen12mscr_dataset import SEN12MSCR

    trainset = SEN12MSCR(opts, root=root_dir, split='train')
    valset   = SEN12MSCR(opts, root=root_dir, split='val')

    N = len(trainset)
    subset_size = max(1, int(frac * N))
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    subset_indices = idx[:subset_size].tolist()
    trainset = Subset(trainset, subset_indices)

    def collate_fn(batch):
        sar    = torch.stack([item['SAR_data']  for item in batch], dim=0)
        opt    = torch.stack([item['cloudy_data']  for item in batch], dim=0)
        target = torch.stack([item['cloudfree_data'] for item in batch], dim=0)
        
        batch_dict = {
            'cloudy_data'       : opt,
            'SAR_data'          : sar,
            'cloudfree_data'    : target
        }
        if 'cloud_mask' in batch[0]:
            batch_dict['cloud_mask'] = torch.stack([item['cloud_mask'] for item in batch], dim = 0)
            
        return batch_dict

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True,
        #generator=shuffle_gen,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset=valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if include_test:
        testset = SEN12MSCR(opts, root=root_dir, split='test')
        test_loader = DataLoader(
            dataset=testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader