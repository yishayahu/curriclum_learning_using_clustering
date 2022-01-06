import torch
import random

class RegularSampler(torch.utils.data.Sampler):
    def __init__(self, data_source:torch.utils.data.Dataset,train_indexes,clustering_indexes,warmup_epochs = 1):
        self.ds = data_source
        self.train_indexes = train_indexes
        self.clustering_indexes = clustering_indexes
        self.warmup_epochs =warmup_epochs
        self._clustering_flag = "training"
        self.counter = 0
    def get_clustering_flag(self):
        return self._clustering_flag
    def __iter__(self):
        if self.counter < self.warmup_epochs+1:
            self.counter+=1
            indexes = self.train_indexes
            random.shuffle(indexes)
            for idx in indexes:
                yield idx
        if self.counter == self.warmup_epochs+1:
            self._clustering_flag = "clustering"
            indexes = self.clustering_indexes
            random.shuffle(indexes)
            for idx in indexes:
                yield idx
            self._clustering_flag = "done"
            for i in range(8):
                yield random.choice(self.clustering_indexes)


    def __len__(self):
        return len(self.ds)