import os
import random

import torch
from sklearn.cluster import MiniBatchKMeans
from clustered_sampler import ClusteredSampler
from regular_sampler import RegularSampler
import numpy as np

class DsWrapper(torch.utils.data.Dataset):
    def __init__(self,model,dataset_creator,n_clusters,start_transform,feature_layer_name,warmups,exp_name,decrease_center,**kwargs):

        #####
        self.future_transform = kwargs.pop("transform")

        kwargs["transform"] = start_transform
        self.ds = dataset_creator(**kwargs)
        self.dataset_creator = dataset_creator
        self.future_kwargs = kwargs
        self.exp_name = exp_name
        self.decrease_center= decrease_center
        #####


        #######
        all_indexes = list(range(len(self.ds)))
        random.shuffle(all_indexes)
        train_indexes = all_indexes[:int(len(self.ds)*0.8)]
        clustering_indexes = all_indexes[int(len(self.ds)*0.8):]

        regular_sampler = RegularSampler(data_source=self, train_indexes=train_indexes,
                                         clustering_indexes=clustering_indexes,warmup_epochs=warmups)
        self.current_sampler = regular_sampler



        ######
        self.new_indexes = []
        self.index_to_cluster = {}
        self.n_clusters = n_clusters
        self.losses = [[] for _ in range(n_clusters)]
        self.clustering_algorithm = MiniBatchKMeans(n_clusters=n_clusters)
        self.arrays = []
        self.indexes = []
        def feature_layer_hook(model, input, output):
            input = input[0]
            if model.training and type(self.current_sampler) == RegularSampler:
                assert input.shape[0] == len(self.new_indexes)

                for i in range(input.shape[0]):
                    self.arrays.append(input[i].cpu().detach().flatten().numpy())
                    self.indexes.append(self.new_indexes[i])

            else:
                if not model.training:
                    assert len(self.new_indexes) == 0
        for module_name, module1 in model.named_modules():
            if module_name == feature_layer_name:
                module1.register_forward_hook(feature_layer_hook)


        #######

    def send_loss(self,loss,train_loader):
        assert loss.shape[0] == len(self.new_indexes)
        if self.current_sampler.get_clustering_flag() == "clustering":
            self.clustering_algorithm.partial_fit(self.arrays)
            for (index, label) in zip(self.indexes, self.clustering_algorithm.predict(self.arrays)):
                self.index_to_cluster[index] = label
            self.arrays = []
            self.indexes = []
            for i, index in enumerate(self.new_indexes):
                self.losses[self.index_to_cluster[index]].append(loss[i].item())
            self.new_indexes = []
            loss[loss!=0] = 0

        elif self.current_sampler.get_clustering_flag() == "done":

            for i, index in enumerate(self.new_indexes):
                self.losses[self.index_to_cluster[index]].append(loss[i].item())
            self.new_indexes = []
            loss[loss!=0] = 0
            self.future_kwargs["transform"] = self.future_transform
            self.ds = self.dataset_creator(**self.future_kwargs)
            assert len(self.index_to_cluster) == len(self.ds)
            self.current_sampler = ClusteredSampler(data_source=self.ds, index_to_cluster=self.index_to_cluster,
                                                    n_cluster=self.n_clusters, losses=self.losses,exp_name=self.exp_name,decrease_center=self.decrease_center)
            train_loader.recreate(dataset=self.ds,sampler=self.current_sampler)
            self.new_indexes = []

        else:
            if len(self.arrays)>=10000:
                self.clustering_algorithm.partial_fit(self.arrays)
                for (index, label) in zip(self.indexes, self.clustering_algorithm.predict(self.arrays)):
                    self.index_to_cluster[index] = label
                self.arrays = []
                self.indexes = []
            self.new_indexes = []
        return torch.mean(loss)


    def __getitem__(self, item):
        if type(self.current_sampler) == RegularSampler:
            self.new_indexes.append(item)
        return self.ds[item]
    def __len__(self):
        return len(self.ds)



