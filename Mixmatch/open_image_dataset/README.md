# Experiment on Open Image Dataset v6 (OIDv6)
## Setup

1. Install [OIDv6 tool](https://github.com/DmitryRyumin/OIDv6) to download specific categories from the whole dataset.
	``` pip install oidv6```
2. Navigate to /Mixmatch/open_image_dataset directory
3. Download some categories from the dataset to current directory. (This may take a while)
	Take Motorcycle, Parking Meter, Stop Sign, Fire Hydrant these 4 categories as an example:
	```oidv6 downloader --dataset . --type_data all --classes "Motorcycle" "Parking meter" "Stop sign" "Fire hydrant" --yes```
4. Run img2tfrecord.py to generate tfRecord files from the downloaded images:
	```python3 img2tfrecord.py```.
	The generated tfrecord files will be saved to Mixmatch/ML_DATA
5. Run script to generate 1000 labeled data and ~7000 unlabeled data to experiment on the semi-supervised learning model. The ratio between labeled data can be modified inside the shell script.
	```.buildOIDv6_dataset.sh```
6. Run the model on this dataset.
	```CUDA_VISIBLE_DEVICES=0 python ../mixmatch.py --filters=32 --dataset=OIDv6.3@1000-1 --w_match=75 --beta=0.75```

## Experiment Dataset Info
| Class         | Label | Train | Validation | Test |
|---------------|-------|-------|------------|------|
| Motorcycle    | 0     | 6944  | 151        | 442  |
| Parking Meter | 1     | 177   | 4          | 5    |
| Stop Sign     | 2     | 375   | 13         | 34   |
| Fire Hydrant  | 3     | 408   | 15         | 60   |

The dataset is biased on class Motorcycle, which dominates ~85% of all the cases. And from all the training data, 1000 data points are labeled, and the rest is unlabeled.

## Results:

The model achieved 91.18% accuracy on the test set, and 100% on the training set on epoch 6. And it quickly overfitted on the dataset.

And these are the PR curves for each category:

**0. Motorcycle:**
![Motorocycle PR Curve](./Results/00_Motorcycle.png?raw=true)

**1. Parking Meter:**
![Parking Meter PR Curve](./Results/01_Parking_Meter.png?raw=true)

**2. Stop Sign:**
![Stop Sign PR Curve](./Results/02_Stop_Sign.png?raw=true)

**3. Fire Hydrant:**
![Fire Hydrant PR Curve](./Results/03_Fire_Hydrant.png?raw=true)
