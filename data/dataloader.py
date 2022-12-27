import torch.utils.data
from torch.utils.data import RandomSampler

from data.dataset import MRImageData


class CombinedImageLoader(object):
    def __init__(self, dataset, modality, transform, batch_size=1, num_workers=4, subject_id=None, aug=None,
                 frame=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert len(dataset) == 2, 'Requires two datasets: source, target'
        assert len(modality) == 2, 'Requires specific modalities of two datasets: source, target'

        self.source = MRImageData(folder=dataset[0], is_supervised=True, modality=modality[0], transform=transform,
                                  subject_id=subject_id[0] if subject_id else None, aug=aug,
                                  frame=frame[0] if frame else None)
        self.target = MRImageData(folder=dataset[1], is_supervised=True, modality=modality[1], transform=transform,
                                  subject_id=subject_id[1] if subject_id else None, aug=aug,
                                  frame=frame[1] if frame else None)
        print('[Source image settings] ', dataset[0], modality[0], subject_id[0])
        print('[Target image settings] ', dataset[1], modality[1], subject_id[1])
        self.loader_src = None
        self.loader_tgt = None
        self.iters_src = None
        self.iters_tgt = None

        self.n = max(len(self.source), len(self.target))  # make sure you see all images
        self.sampler_src = RandomSampler(self.source, replacement=True, num_samples=self.n)
        self.sampler_tgt = RandomSampler(self.target, replacement=True, num_samples=self.n)
        print('[Data Loader] Source length:', len(self.source), 'Target length:', len(self.target))

        self.num = 0
        self.set_loader_src()
        self.set_loader_tgt()

    def __len__(self):
        return self.n

    def set_loader_src(self):
        collate_fn = torch.utils.data.dataloader.default_collate
        self.loader_src = torch.utils.data.DataLoader(
            self.source, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True,
            sampler=self.sampler_src, collate_fn=collate_fn, pin_memory=False, prefetch_factor=2
        )
        self.iters_src = iter(self.loader_src)

    def set_loader_tgt(self):
        collate_fn = torch.utils.data.dataloader.default_collate
        self.loader_tgt = torch.utils.data.DataLoader(
            self.target, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True,
            sampler=self.sampler_tgt, collate_fn=collate_fn, pin_memory=False, prefetch_factor=2
        )
        self.iters_tgt = iter(self.loader_tgt)

    def __iter__(self):
        for i, data in enumerate(zip(self.loader_src, self.loader_tgt)):
            if i * self.batch_size >= self.n:
                break
            yield *data[0], *data[1]
