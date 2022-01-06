import torch
class DataLoaderWrapper(torch.utils.data.DataLoader):
    def __init__(self,dataloader_creator):
        self.dataloader_creator = dataloader_creator
        self.dataloader_kwargs = {}
    def recreate(self,**dataloader_kwargs):
        for key,val in dataloader_kwargs.items():
            self.dataloader_kwargs[key] = val
        self.dl = self.dataloader_creator(**self.dataloader_kwargs)
        return self
    @property
    def _dataset_kind(self):
        return self.dl._dataset_kind

    @property
    def batch_sampler(self):
        return self.dl.batch_sampler
    @property
    def persistent_workers(self):
        return self.dl.persistent_workers
    @property
    def num_workers(self):
        return self.dl.num_workers

    @property
    def dataset(self):
        return self.dl.dataset

    @property
    def _IterableDataset_len_called(self):
        return self.dl._IterableDataset_len_called

    @property
    def drop_last(self):
        return self.dl.drop_last

    @property
    def prefetch_factor(self):
        return self.dl.prefetch_factor

    @property
    def _prefetch_factor(self):
        return self.dl._prefetch_factor
    @property
    def pin_memory(self):
        return self.dl.pin_memory
    @property
    def timeout(self):
        return self.dl.timeout
    @property
    def collate_fn(self):
        return self.dl.collate_fn

    @property
    def generator(self):
        return self.dl.generator
    @property
    def sampler(self):
        return self.dl.sampler
