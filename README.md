# Image Captioning based on Bottom-Up and Top-Down Attention model

Our overall approach centers around the Bottom-Up and
Top-Down Attention model, as designed by [Anderson et al](https://arxiv.org/pdf/1707.07998v3.pdf). We used this framework as a starting point for further experimentation, implementing, in addition to various hy-
perparameter tunings, two additional model architectures.
First, we reduced the complexity of Bottom-Up and Top-
Down by considering only a simple LSTM architecture.
Then, taking inspiration from the Transformer architecture,
we implement a non-recurrent model which does not need
to keep track of an internal state across time. Our results are
comparable to the author’s implementation of the Bottom-
Up and Top Down Attention model. Our code serves as a
baseline for future experiments which are done using the
Pytorch framework.

## Getting Started

Machine configuration used for testing: Nvidia P40 GPUs card with 24GB memory (though a machine with lesser memory would work just fine)

We use the Karpathy splits as described in [Deep visual-semantic alignments for generating image descriptions.](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf). The Bottom-Up image features are used directly from [here](https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip). Please refer to [this repo](https://github.com/peteanderson80/Up-Down-Captioner) for clarifications. The annotations are downloaded from the [COCO website](http://cocodataset.org/#download) (2014 train val annotations). All the models have been trained from scratch.

The code takes around 8 hours to train on the karpathy train split.

### Prerequisites

What things you need to install the software and how to install them

Software used:
1. Pytorch 0.4.1
2. Python 2.7

Dependencies:

If you are not using conda as a package manager, refer to the yml file and install the libraries manually.

## Running the code

Run main.sh script with the appropriate arguments. The arguments have been listed in the [src/utils_experiment.py](https://github.com/njchoma/transformer_image_caption/blob/master/src/utils_experiment.py) file. 

After the model has been trained, run [src/evaluate_test.py](https://github.com/njchoma/transformer_image_caption/blob/master/src/evaluate_test.py)

## License

## Contributors

1. Nicholas Choma (New York University)
2. Omkar Damle (New York University)

This code was produced as a part of my course project at New York University. I would like to thank Prof. Fergus for his guidance and providing access to the GPUs.

References:
1. Code for metrics evaluation was borrowed from https://github.com/tylin/coco-caption
