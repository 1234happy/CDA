# The testing codes and pretrained models for protocols I&M to O, C&M to O, and I&C to O.

## Requirements
- python 3.6 
- pytorch 0.4 
- torchvision 0.2
- cuda 8.0

## Pre-testing

**Dataset.** 

Download the OULU-NPU datasets.

**Data Pre-processing.** 

[MTCNN algotithm](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) is used for face detection and face alignment. All the detected faces are normlaize to 224$\times$224$\times$3. 

Generate the data label list for all the detected faces. 
The labels for live face images and spoof face images are 1 and 0, respectively.
Save the list to a .txt file.


## Testing

Run like this:
```python
python test.py --data_path_test ./data/datapath.txt --resume ./checkpoint/oulu/C_M2O/checkpoint.pth.tar

```

Some evaluation codes are from [SSDG](https://github.com/taylover-pei/SSDG-CVPR2020).





