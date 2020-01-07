# Background Remover

## Overview

To remove objects from images, there are several algorithms:

* **Clustering**
    * It usually partition the image into several clusters.
    * K-means is a well known method.
* **Thresholding**
    * The simplest method.
    * The key is to select a threshold value and then compare to each pixel.
* **Region Growing**
    * Mainly relies on the that the neighbors in same region should be similar.
* **Deep Learing**
    * It has an enormous achievement on this field.
    * Usually be implemented with convolutional layers.

All of them are very powerful and interesting, but we'll implement the remover with **deep learing**. 

## CNN (Convolutional Neural Network)

In deep learning, tasks about image are often solved with **CNN**. <br>
CNN has some powerful benefits:
* It takes important features from images, such as edges.
* In deep learning, it reduces the number of parameters, but has better performance.
* Network can be calculated on GPUs more faseter than on CPUs.

## Model Explain

There are many models for image segmentation made by well known organizations and researchers. <br>
We'll use **U-Net** in this example.

### U-Net

In traditional models, layers are usually connected to the next one. <br>
While more maxpooling layers inputs go through, the more features are lost.

U-Net solve this problem in a clever way. <br>
<img src="https://img-blog.csdn.net/20181022150306666?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dpdGh1Yl8zNjkyMzQxOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" width="70%">

It add outputs from encoder to layers of decoder directly, so the decoder can use more details.

## Prepare Dataset

Before building model, we should prepare our data first. <br>
We'll use images from **COCO dataset** to train the model.

Download train, validation and test dataset.

```
!wget http://images.cocodataset.org/zips/train2017.zip
!wget http://images.cocodataset.org/zips/val2017.zip
!wget http://images.cocodataset.org/zips/test2017.zip
```

Extract all datasets.

```
!unzip train2017.zip
!unzip val2017.zip
!unzip test2017.zip
```

To use COCO dataset for training, we need **annotation** files to get masks of segmentation.

```
!wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
!unzip annotations_trainval2017.zip
```

Now, you should have three datasets and several json files in annotation folder. <br>
Next, we have to preprocess the images by creating mask images.

Load data information from annotations.

```python
from pycocotools.coco import COCO
# annotations/instances_{dataset}2017.json
path = join(home, "annotations/instances_train2017.json")
data = COCO(path)
```

We only need images that ccontain person.

```python
import numpy as np
import cv2
from os.path import join

i = 0
images, masks = [], []
for img in data.loadImgs(data.getImgIds()):
    
    valid = False
    anns = data.loadAnns(data.getAnnIds(imgId))

    mask = np.zeros((img["height"], img["width"]), dtype=np.byte)
    for ann in anns:
        
        # category id of person is 1
        if ann["category_id"] == 1:
            seg = data.annToMask(ann)
            mask += seg
            valid = True
    
    # if contains person
    if valid:
        
        file_name = img["file_name"]
        # {dataset}2017
        frame = cv2.imread("train2017" + file_name)
        # frames/{dataset}
        cv2.imwrite(join("frames/train", file_name), image)
        # masks/{dataset}
        cv2.imwrite(join("masks/train", file_name), mask)
        i += 1
```

## Build Model

In this example, we'll use 16000 samples for training.

```python
# input image shape
img_shape = (256, 256)
batch_size = 32
num_train = 500 * batch_size
num_val = 50 * batch_size
```

```python
import cv2
import glob
from os.path import join, expanduser
import numpy as np

# define our own pipeline
def generator(frames_path, masks_path, batch_size, img_shape, num_data=None):
    frames_path = glob.glob(join(frames_path, "*.jpg"))
    frames_path = sorted(frames_path, key=lambda path: int(path.split('/')[-1].split('.')[0]))
    masks_path = glob.glob(join(masks_path, "*.jpg"))
    masks_path = sorted(masks_path, key=lambda path: int(path.split('/')[-1].split('.')[0]))

    # use all samples
    if num_data is None:
        num_data = len(frames_path)
    order = np.arange(num_data)
    np.random.shuffle(order)

    base = 0
    while True:

        if base == num_data - 1:
            np.random.shuffle(order)
            base = 0

        frames, masks = [], []
        for i in range(batch_size):
            
            # this ensure that we get correct number of samples
            idx = order[(base + i) % num_data]
            
            frame = cv2.imread(frames_path[idx]).astype(np.float32)
            frame = cv2.resize(frame, img_shape)
            mask = cv2.imread(masks_path[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask = cv2.resize(mask, img_shape).reshape(img_shape + (1, ))

            frames.append(frame)
            masks.append(mask)

        base += batch_size

        # yield makes function iterateble
        yield np.array(frames), np.array(masks)
        del frames, masks, frame, mask, idx
```

```python
train_frames_path = "frames/train"
train_masks_path = "masks/train"
train_generator = generator(train_frames_path, train_masks_path, batch_size, img_shape, num_train)

val_frames_path = "frames/val"
val_masks_path = "masks/val"
val_generator = generator(val_frames_path, val_masks_path, batch_size, img_shape, num_val)
```
