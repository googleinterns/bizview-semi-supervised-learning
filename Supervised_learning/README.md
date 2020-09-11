# Resnet50

The supervised learning model (Resnet50) is used for a baseline comparison with the performance of the Mixmatch model.
The model utilized transfer learning that is pretrained on 'Imagenet', and further tuned on the streetview dataset.

## Results:
Here are the results trained on streetview dataset. For more info on the dataset, reference to the README under Mixmatch/streetview_dataset.

| Dataset           | Labels | Image Size | Learning Rate | Batch Size | Dropout | Pretrained | Accuracy |
|-------------------|--------|------------|---------------|------------|---------|------------|----------|
| streetview_v3_64  | Full   | 64x64      | 0.0001        | 64         | 0.2     | Yes        | 78.1     |
| streetview_v3_64  | 1000   | 64x64      | 0.0001        | 64         | 0.2     | Yes        | 0.679    |
| streetview_v3_64  | 250    | 64x64      | 0.0001        | 64         | 0.2     | Yes        | 0.523    |
| streetview_v3_256 | Full   | 256x256    | 0.0001        | 64         | 0.2     | Yes        | 86.4     |
| streetview_v3_256 | Full   | 256x256    | 0.0001        | 8          | 0.2     | Yes        | 84.48    |
| streetview_v4_64  | Full   | 64x64      | 0.0001        | 64         | 0.2     | No         | 73.8     |