import os
from data import init_dataloader  
from models.resnet import resnet  
import numpy as np
import argparse
import torch
from sklearn.metrics import roc_curve, auc
import math
def main(args):

    model = resnet(args)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    test_loader = init_dataloader.get_loader(args)
    model.eval()
    prob_list=[]
    label_list=[]
    pos_label=1

    for i, (input, target,imagenames) in enumerate(test_loader):
        input_var = torch.autograd.Variable(input) 
        outputs = model(input_var)
        for i in range(len(imagenames)):
            prob_batch=torch.nn.functional.softmax(outputs, dim=1).data[i]
            prob_list.append(float(prob_batch[1])+float(prob_batch[1+args.num_classes])+float(prob_batch[1+2*args.num_classes]))  
            label_list.append(int(target[i]))  

    fpr, tpr, thresholds= roc_curve(label_list, prob_list,pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    cur_EER, threshold, _, _ = get_EER_states(prob_list, label_list)
    cur_HTER = get_HTER_at_thr(prob_list, label_list, threshold)
    print('ROC: {}, HTER: {}'.format(roc_auc,cur_HTER))


def eval_state(probs, labels, thr):
    predict=[]
    for i in range(len(probs)):
        if probs[i]>=thr:
            predict.append(True)
        else:
            predict.append(False)
    TN=0
    FN=0
    FP=0
    TP=0
    for i in range(len(probs)):
        if (labels[i] == 0) & (predict[i] == False):
            TN+=1
        if (labels[i] == 1) & (predict[i] == False):
            FN+=1   
        if (labels[i] == 0) & (predict[i] == True):
            FP+=1
        if (labels[i] == 1) & (predict[i] == True):
            TP+=1
    return TN, FN, FP, TP
def get_threshold(probs, grid_density):
    Min, Max = min(probs), max(probs)
    thresholds = []
    for i in range(grid_density + 1):
        thresholds.append(0.0 + i * 1.0 / float(grid_density))
    thresholds.append(1.1)
    return thresholds
def get_EER_states(probs, labels, grid_density = 10000):
    thresholds = get_threshold(probs, grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if(FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif(FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    thr = min_dist_states[2]
    return EER, thr, FRR_list, FAR_list

def get_HTER_at_thr(probs, labels, thr):
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    if (FN + TP == 0):
        FRR = 1.0
        FAR = FP / float(FP + TN)
    elif(FP + TN == 0):
        FAR = 1.0
        FRR = FN / float(FN + TP)
    else:
        FAR = FP / float(FP + TN)
        FRR = FN / float(FN + TP)
    HTER = (FAR + FRR) / 2.0
    return HTER

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train args',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path_test', type=str, default='path.txt',
                        help='file path of the testing images')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--arch', type=str, default='resnet18', help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing.')
    parser.add_argument('--resume', type=str, default='./checkpoint/oulu/C_M2O/checkpoint.pth.tar', help='Checkpoints path')
    args = parser.parse_args()

    main(args)

