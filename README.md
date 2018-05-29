# Image Captioning

## Introduction

Build a model to generate captions from images. When given an image, the model is able to describe in English what is in the image. In order to achieve this, our model is comprised of an **encoder** which is a CNN and a **decoder** which is an RNN. The CNN encoder is given images for a classification task and its output is fed into the RNN decoder which outputs English sentences.

The model and the tuning of its hyperparamaters are based on ideas presented in the paper [**Show and Tell: A Neural Image Caption Generator**](https://arxiv.org/pdf/1411.4555.pdf) and [**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**](https://arxiv.org/pdf/1502.03044.pdf).

We use the Microsoft **C**ommon **O**bjects in **CO**ntext (MS COCO) [dataset](http://cocodataset.org/#home) for this project. It is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms. For instructions on downloading the data, see the **Data** section below.

## Code

The code can be categorized into two groups:

1) Notebooks - The main code for the project is structured as a series of Jupyter notebooks:

* `0_Dataset.ipynb` - Introduces the dataset and plots some sample images.
* `1_Preliminaries.ipynb` - Loads and pre-processes data and experiments with models.
* `2_Training.ipynb` - Trains a CNN-RNN model.
* `3_Inference.ipynb`- Generates captions for test images.

2) Helper files - Contain helper code for the notebooks:
* `data_loader.py`- Creates the `CoCoDataset` and a DataLoader for it.
* `vocabulary.py` - Tokenizes captions and adds them to a dictionary of vocabulary. It is used as an instance variable of the `CoCoDataset`.
* `model.py` - Provides the CNN and RNN models that are used by the notebooks to train and test data.

## Setup

1. Clone the [COCO API repo](https://github.com/cocodataset/cocoapi) into *this project's directory*:
```
git clone https://github.com/cocodataset/cocoapi.git
```

2. Setup COCO API (also described in the readme [here](https://github.com/cocodataset/cocoapi)):
```
cd cocoapi/PythonAPI
make
cd ..
```

3. Install PyTorch (4.0 recommended) and torchvision.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install -c peterjc123 pytorch-cpu
	pip install torchvision
	```

4. Others:

* Python 3
* pycocotools
* nltk
* numpy
* scikit-image
* matplotlib
* tqdm

## Data

Download the following data from the [COCO website](http://cocodataset.org/#download), and place them, as instructed below, into the `cocoapi` subdirectory located *inside* this project's directory (the subdirectory was created when cloning the COCO API repo as shown in the **Setup** section above):

* under **Annotations**, download:
  - **2014 Train/Val annotations [241MB]** (extract `captions_train2014.json`, `captions_val2014.json`, `instances_train2014.json` and `instances_val2014.json`, and place them in the subdirectory `cocoapi/annotations/`)
  - **2014 Testing Image info [1MB]** (extract `image_info_test2014.json` and place it in the subdirectory `cocoapi/annotations/`)
* under **Images**, download:
  - **2014 Train images [83K/13GB]** (extract the `train2014` folder and place it in the subdirectory `cocoapi/images/`)
  - **2014 Val images [41K/6GB]** (extract the `val2014` folder and place it in the subdirectory `cocoapi/images/`)
  - **2014 Test images [41K/6GB]** (extract the `test2014` folder and place it in the subdirectory `cocoapi/images/`)
          
## Run

To run any script file, use:

```bash
python <script.py>
```

To run any IPython Notebook, use:

```bash
jupyter notebook <notebook_name.ipynb>
```
