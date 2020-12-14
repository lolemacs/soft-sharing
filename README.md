[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-implicitly-recurrent-cnns-through/architecture-search-on-cifar-10-image)](https://paperswithcode.com/sota/architecture-search-on-cifar-10-image?p=learning-implicitly-recurrent-cnns-through)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-implicitly-recurrent-cnns-through/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=learning-implicitly-recurrent-cnns-through)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-implicitly-recurrent-cnns-through/image-classification-on-cifar-100)](https://paperswithcode.com/sota/image-classification-on-cifar-100?p=learning-implicitly-recurrent-cnns-through)


# Soft Parameter Sharing

Author implementation of the soft sharing scheme proposed in "Learning Implicitly Recurrent CNNs Through Parameter Sharing"  [[PDF](https://openreview.net/pdf?id=rJgYxn09Fm)]

[Pedro Savarese](https://ttic.uchicago.edu/~savarese), [Michael Maire](http://ttic.uchicago.edu/~mmaire/)

Soft sharing is offered as stand-alone PyTorch modules (in models/layers.py), which can be used in plug-and-play fashion on virtually any CNN.





## Requirements
```
Python 2, PyTorch == 0.4.0, torchvision == 0.2.1
```
The repository should also work with Python 3.

[BayesWatch's ImageNet Loader](https://github.com/BayesWatch/sequential-imagenet-dataloader) is required for ImageNet training.





## Using soft parameter sharing

The code in models/layers.py offers two modules that can be used to apply soft sharing to standard convolutional layers: TemplateBank and SConv2d (shared 2d convolution).

You can take any model that is defined using standard Conv2d:

```
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=0)
    
    self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
    
  def forward(self, x):
    x = F.relu(self.conv1(x)))
    x = F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x)))
    x = F.relu(self.conv4(x)))
    return x
```

And, to apply soft sharing among the convolutional layers, first create a TemplateBank and replace Conv2d layers by SConv2d, passing the created bank as first argument.

```
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=0)
    
    self.bank = TemplateBank(num_templates=3, in_planes=10, out_planes=10, kernel_size=3)
    self.conv2 = SConv2d(bank=self.bank, stride=1, padding=1)
    self.conv3 = SConv2d(bank=self.bank, stride=1, padding=1)
    self.conv4 = SConv2d(bank=self.bank, stride=1, padding=1)
    
  def forward(self, x):
    x = F.relu(self.conv1(x)))
    x = F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x)))
    x = F.relu(self.conv4(x)))
    return x
```

Make sure not to apply weight decay to the coefficients for best results (check how the group_weight_decay() function is used for this purpose in main.py).


## Training the model

To train a SWRN-28-10-6 with cutout on CIFAR-10, do:

```
python main.py data --dataset cifar10 --arch swrn --depth 28 --wide 10 --bank_size 6 --cutout --job-id swrn28-10-6
```

By default the learning rate will be decayed by 5 at epochs 60, 120 and 160 (out of a total of 200 epochs), and a weight decay of 0.0005 is applied. These settings can be specified through command-line arguments (schedule, gammas, decay, etc).

Also by default a 90/10 split will be used to split the original training set into train/val. When your model is ready to be evaluated on the test set, you can use the --evaluate option and point to the saved model, as in:

```
python main.py data --dataset cifar10 --arch swrn --depth 28 --wide 10 --bank_size 6 --evaluate --resume snapshots/swrn28-10-6/model_best.pth.tar
```



## Citation

```
@inproceedings{
savarese2018learning,
title={Learning Implicitly Recurrent {CNN}s Through Parameter Sharing},
author={Pedro Savarese and Michael Maire},
booktitle={International Conference on Learning Representations},
year={2019}
}
```

