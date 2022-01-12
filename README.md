# curriclum_learning_using_clustering
## reproduce CIFAR-10 accuracy using UPANet
```bash
cd UPANets
python main.py
python main.py  --use_clustering_curriculum
```
## reproduce Tiny ImageNet using Densenet
download Tiny ImageNet dataset
```bash
python main.py path/to/tiny/ImageNet -a Densenet121 --exp_name baseline
python main.py path/to/tiny/ImageNet --use_clustering_curriculum -a Densenet121 --exp_name clustering
```

## reproduce MRI skull stripping using UNET-2D
The code to reproduce this expirement is in https://github.com/yishayahu/domain_shift_anatomy.git.
