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

[MTCNN algotithm](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) is used for face detection and face alignment. All the detected faces are normlaize to 224\times224\times3. 

Generate the data label list for all the detected faces like this:
$root/data/spoof1.jpg 0  
$root/data/live1.jpg 1  
...  
$root/data/spoofN.jpg 0  
$root/data/liveN.jpg 1  

Save the list to a .txt file.


## Testing

Run like this:
```python
python test.py --data_path_test ./data/datapath.txt --resume ./checkpoint/oulu/C_M2O/checkpoint.pth.tar

```

Some evaluation codes are from [SSDG](https://github.com/taylover-pei/SSDG-CVPR2020).





