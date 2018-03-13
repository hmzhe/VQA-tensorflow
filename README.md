# VQA-tensorflow-tensorlayer

This is a tensorflow/tensorlayer implementation for VQA, MLP+Image-Question-Co-attention, model.
The MLP part is explained in the paper [Revisiting visual question answering baselines](https://arxiv.org/pdf/1606.08390.pdf).
The Image-question-co-attention part is explained in the paper [Hierarchical question-image co-attention for visual question answering](https://arxiv.org/pdf/1606.00061v1.pdf)

Details about the dataset are explained at the [Visual7W](web.stanford.edu/~yukez/visual7w/). 

Here is a summary of performance we obtained on both the models.

| Model            | Epochs | Batch Size | Validation Accuracy |
|------------------|--------|------------|---------------------|
| MLP+Co-Attention | 50     | 32         | 60.32%              |

## Requirements

* Python 3.6
* Tensorflow
* Tensorlayer
* Numpy
