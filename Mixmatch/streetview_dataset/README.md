# Experiment on Streetview Storefront Imagery
## Dataset Info
The dataset in the experiment is a combination of [TC11 Streetview dataset](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)  and [UCF Google Streeview dataset](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/).
The UCF dataset will be passed through a storefront detector that creates bounding boxes(bbox) with confidence score of having a storefront inside that area. This information will be added into the creation of datasets.

There are 4 versions of streetview dataset that are used throughout the experiment:
- **streetview_v1:** This dataset mixes the TC11 and UCF images. For the UCF, only the **MAX** confidence bbox over threshold 0.9 will be cropped and added into dataset. For the TC11, it will be added as positive examples for both train and test set.
- **streetview_v2:** This is a mixture of TC11 and UCF dataset. Compared to streetview_v1, **ALL** bbox over threshold will be cropped and added into dataset, instead of only cropping the highest bbox in a image. Also, from this version two views of UCF are removed, there is no marked view and skyview images included in the dataset. (More info of the views can be found in the UCF dataset website)
- **streetview_v3:** It's similar to streetview_v2. The difference are:
    1. Trainset is balanced, which means positive and negative examples are equal in quantity.
    2. The trainset only contains data from UCF.
    3. Testset is a combination of TC11 as positive examples and handpick UCF with confidence < 0.2 as negative examples.
- **streetview_v4:** It's streetview_v3 with an addition of word embeddings added to the dataset. The word embedding are generated from spacy.io word2vec API using the texts generated from Google OCR API using the images.
![Streetview_v4 word embeddings](./IMG/streetview_v4_data_processing.png?raw=true)

| Dataset Version | Training Set Count (Pos/Neg Examples) | Testing Set Count (Pos/Neg Examples) 
|-----------------|-------|-------|
| streetview_v1   | 21,124 (14,132 / 6,992)     | 2,224 (1,517 / 707) |
| streetview_v2   | 296,398 (25,100 / 271,298)  | 31,963 (2,640 / 29,323)   |
| streetview_v3   | 39,774 (19,887 / 19,887)    | 754 (350 / 404)   |
| streetview_v4   | 39,774 (19,887 / 19,887)    | 754 (350 / 404)   |

## Results:
Thorough experiments are done on streetview_v3 and streetview_v4 datasets. Here are the results:
### Streetview_v3:
#### **Label Size to Accuracy**
The label size is the amount of labeled data within the training set.

|            Dataset           | Labels | Learning Rate | Batch Size | Accuracy |
|:----------------------------:|:------:|:-------------:|:----------:|:--------:|
|   streetview_v3_64.1@25-200  |   25   |     0.0001    |     64     |   57.56  |
|  streetview_v3_64.1@100-200  |   100  |     0.0001    |     64     |   67.24  |
|  streetview_v3_64.1@250-200  |   250  |     0.0001    |     64     |   76.79  |
|  streetview_v3_64.1@1000-200 |  1000  |     0.0001    |     64     |   71.62  |
|  streetview_v3_64.1@8000-200 |  8000  |     0.0001    |     64     |   85.94  |
| streetview_v3_64.1@20000-200 |  20000 |     0.0001    |     64     |   70.03  |
| streetview_v3_64.1@39000-200 |  39000 |     0.0001    |     64     |   79.97  |

![Label Size to Accuracy](./IMG/streetview_v3_pr_curve.png?raw=true)


---
#### **Learning Rate to Accuracy**
|            Dataset           | Labels | Learning Rate | Batch Size | Accuracy |
|:----------------------------:|:------:|:-------------:|:----------:|:--------:|
|  streetview_v3_64.1@250-200  |   250  |    0.00005    |     64     |  77.19%  |
|  streetview_v3_64.1@250-200  |   250  |     0.0001    |     64     |  76.66%  |
|  streetview_v3_64.1@250-200  |   250  |     0.0005    |     64     |  78.65%  |
|  streetview_v3_64.1@250-200  |   250  |     0.001     |     64     |  74.80%  |
|  streetview_v3_64.1@250-200  |   250  |     0.002     |     64     |  78.91%  |
| streetview_v3_64.1@20000-200 |  20000 |     0.0001    |     64     |   70.03  |
| streetview_v3_64.1@39000-200 |  39000 |     0.0001    |     64     |   79.97  |

![Learning Rate to Accuracy](./IMG/streetview_v3_learning_rate_pr.png?raw=true)

---
#### **Model Layer Depth to Accuracy**
The layer depth is the numbers of layers within the Mixmatch residual network.

|           Dataset          | Learning Rate | Batch Size | Layers (Depth) | Training Speed | Accuracy |
|:--------------------------:|:-------------:|:----------:|:--------------:|:--------------:|:--------:|
| streetview_v3_64.1@250-200 |     0.0001    |     64     |       46       |      300%      |   76.62  |
| streetview_v3_64.1@250-200 |     0.0001    |     64     |       78       |      180%      |   79.44  |
| streetview_v3_64.1@250-200 |     0.0001    |     64     |       110      |      130%      |   79.84  |
| streetview_v3_64.1@250-200 |     0.0001    |     64     |       142      |      100%      |   78.65  |
| streetview_v3_64.1@250-200 |     0.0001    |     64     |       174      |       81%      |   80.37  |
| streetview_v3_64.1@250-200 |     0.0001    |     64     |       206      |       69%      |   78.38  |

![Model Layer Depth to Accuracy](./IMG/streetview_v3_depth_size_pr.png?raw=true)

---
#### **Image Size to Accuracy**
All the previous results used a 64x64 size dataset. Here we test using 256x256 images.
NOTE: Due to machine limitation, on 256x256 dataset the batch size is shrieked from 64 to 8. Thus, adding a control group on 64x64 with batch size 8 for comparison.

|           Dataset           | Image Size | Labels | Learning Rate | Batch Size | Accuracy |
|:---------------------------:|:----------:|:------:|:-------------:|:----------:|:--------:|
| streetview_v3_256.1@250-200 |   256x256  |   250  |     0.0001    |      8     |   68.17  |
|  streetview_v3_64.1@250-200 |    64x64   |   250  |     0.0001    |     64     |   76.66  |
|  streetview_v3_64.1@250-200 |    64x64   |   250  |     0.0001    |      8     |   67.77  |


### Streetview_v4:
There are two ways to add the word embedding to the model, one way is to concatenate the vectors after the image pixels, the other way is to feed them into the model at the top fully connected layers.
![Model Layer Depth to Accuracy](./IMG/streetview_v4_embedding.png?raw=true)

| Label Size | Learning Rate | Batch Size | Accuracy w/o word embeddings | Accuracy w/ word embeddings A method | Accuracy w/ word embeddings B method |
|:----------:|:-------------:|:----------:|:----------------------------:|:------------------------------------:|:------------------------------------:|
|     250    |     0.0001    |     64     |             76.79            |                 74.8                 |                 77.85                |
|    1000    |     0.0001    |     64     |             71.62            |                 80.24                |                 80.11                |
|    8000    |     0.0001    |     64     |             85.94            |                 83.69                |                 81.43                |
|    20000   |     0.0001    |     64     |             70.03            |                 85.41                |                 79.84                |
|    39000   |     0.0001    |     64     |             79.97            |                 84.48                |                 86.74                |