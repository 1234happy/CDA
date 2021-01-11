# The testing codes and pretrained models which are trained on protocols I&M to O, C&M to O, and I&C to O.

## Requirements
- python 3.6 
- pytorch 0.4 
- torchvision 0.2
- cuda 8.0

## Pre-testing

**Dataset.** 

Download the OULU-NPU, CASIA-FASD, Idiap Replay-Attack, and MSU-MFSD datasets.

**Data Pre-processing.** 

[MTCNN algotithm](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) is utilized for face detection and face alignment.
All the detected faces are normlaize to 224$\times$224$\times$3. Only RGB channels are utilized for training. 
Generate the data label list for all the detected faces. The label for live face images is 1, and the label for spoof face images is 0.
Save the list to a .txt file.


## Testing

Run like this:
```python
python test.py --data_path_test ./data/datapath.txt --resume ./checkpoint/oulu/C_M2O/checkpoint.pth.tar

```

Some evaluation codes are from https://github.com/taylover-pei/SSDG-CVPR2020.





