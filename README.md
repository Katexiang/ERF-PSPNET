# ERF-PSPNET
ERF-PSPNET implemented by tensorflow
## Paper
The code is impledmented according to the following papers.
+ [Unifying Terrain Awareness for the Visually Impaired through Real-Time Semantic Segmentation](https://www.mdpi.com/1424-8220/18/5/1506 )
+ [Unifying terrain awareness through real-time semantic segmentation](http://www.wangkaiwei.org/file/publications/iv2018_kailun.pdf )

## Main Dependencies
```
tensorflow 1.11
Open CV
Python 3.6.5
```

## Description
This repository serves as a real-time semantic segmentation networks, which is designed for the assistance for visually-impaired people. The code not only implements the tensorflow-version erf-pspnet, but also implements the code for mIOU calculation which haven't been found in other tf-version's code. Our code combines the training and evaluating stages together, which records every claesses' ioU after one epoch, and fulfill visualization supervision during training.

## Useage
The useage is very easy, you only need to download the code, and create a file folder named ***dataset***, and create another four file folders under the dataset named ***train, trainannot, val, valannot*** ,and put into the data we want to train or eval like the example in the repository.

## Note
In the future, I will update the demo code for video, a improved version network and loss-function like IAL for specific tasks.
