# Fashion-MNIST with over 95% accuracy
This architecture is basically a densenet adapted from the Huang et al. paper

Here are the configurations that are tried and the accuracies achieved.

1. 95.21%--->Densenet(k=12,L=100,epochs=175,Random crop, random horizontal flip,cross_entropy)
2. 94.13%--->Densenet(k=12,L=100,epochs=175,Random crop, RandomRotation, RandomVerticalFlip)
3. 94.86%--->Densenet(k=15,L=100,epochs=175,Random crop, random horizontal flip,cross_entropy)

All the configurations have bottleneck and reduction 0.5(Densenet-BC) and SGD with variable learning rate.

The 1st and 2nd each took about 4hrs and 6GB on a 1080Ti and the 3rd took about 5.8hrs.

Please check the results for the check points, summary, loss and error% (accuracy =100-error%) of individual models. 

Training until 300 epochs can improve results.


# Reference

```
@article{Huang2016Densely,
  author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
  title = {Densely Connected Convolutional Networks},
  journal = {arXiv preprint arXiv:1608.06993},
  year = {2016}
}
```
