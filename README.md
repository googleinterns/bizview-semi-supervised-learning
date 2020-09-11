# BizView Semi-Supervised Learning Project

**This is not an officially supported Google product.**

This repository contains the BizView Semi-Supervised Learning Project, which is a Google open source intern project that experiments semi-supervised learning models on Street View imagery.

The Semi-Supervised Learning (SSL) Project is a proof of concept of applying semi-supervised learning models on street view imagery.
The chosen SSL model is [Mixmatch](https://github.com/google-research/mixmatch), there are two dataset experiments on the model:
- Open Image dataset
- Binary Storefront Image dataset

The results of both datasets can be either found under Mixmatch/open_image_dataset and Mixmatch/streetview_dataset

Also, there is a Resnet50 supervised model that act as the baseline model for performance comparison of the SSL model.
The code could be found under Supervised_learning/


## Source Code Headers

Every file containing source code must include copyright and license
information. This includes any JS/CSS files that you might be serving out to
browsers. (This is to help well-intentioned people avoid accidental copying that
doesn't comply with the license.)

Apache header:

    Copyright 2020 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
