import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from models import CEDN,CEDN_small,HED,HED_small
import csv
import os
import h5py
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

def calculate_metrics(pred,gt):
    tp=0
    fp=0
    tn=0
    fn=0
    for img in range(pred.shape[0]):
        for i in range(pred.shape[1]):
            for j in range(pred.shape[2]):
                pred_pixel = pred[img,i,j]
                gt_pixel = gt[img,i,j]
                if pred_pixel==1 and gt_pixel==1:
                    tp+=1
                elif pred_pixel==0 and gt_pixel==0:
                    tn+=1
                elif pred_pixel==0 and gt_pixel==1:
                    fn+=1
                elif pred_pixel==1 and gt_pixel==0:
                    fp+=1
                else:
                    print("error")
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    try:
        precision = tp/(tp+fp)
    except:
        precision = 0
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    f_score = tp/(tp+0.5*(fp+fn))
    return accuracy,precision,specificity,sensitivity,f_score

def find_best_metrics(pred,gt,channel=0):
    f_score = []
    accuracy = []
    precision = []
    specificity = []
    sensitivity = []
    for cutoff in np.arange(0.05,0.95,0.05):
        thresh = pred>cutoff
        acc,pre,spe,sen,f = calculate_metrics(thresh[:,:,:,channel],gt[:,:,:,channel])
        f_score.append(f)
        accuracy.append(acc)
        precision.append(pre)
        specificity.append(spe)
        sensitivity.append(sen)
    return accuracy,precision,specificity,sensitivity,f_score

#load trained models
hed_small_model = HED_small()
hed_small_model.load_weights('Trained Models/hed_small_boa.h5')
hed_model = HED()
hed_model.load_weights('Trained Models/hed_boa.h5')
cedn_small_model = CEDN_small()
cedn_small_model.load_weights('Trained Models/cedn_small_boa.h5')
cedn_model = CEDN()
cedn_model.load_weights('Trained Models/cedn_boa.h5')

#load testing data
X = np.load('data/boa_testing_input.npy')
Y = np.load('data/boa_testing_output.npy')

#make predictions
hed_small_model_pred = hed_small_model.predict(X)
hed_model_pred = hed_model.predict(X)
cedn_small_model_pred = cedn_small_model.predict(X)
cedn_model_pred = cedn_model.predict(X)
gt = Y

#calculate CHL best metrics
channel=0
hed_small_ac,hed_small_pr,hed_small_sp,hed_small_se,hed_small_fs=find_best_metrics(hed_small_model_pred,gt,channel)
hed_ac,hed_pr,hed_sp,hed_se,hed_fs=find_best_metrics(hed_model_pred,gt,channel)
cedn_small_ac,cedn_small_pr,cedn_small_sp,cedn_small_se,cedn_small_fs=find_best_metrics(cedn_small_model_pred,gt,channel)
cedn_ac,cedn_pr,cedn_sp,cedn_se,cedn_fs=find_best_metrics(cedn_model_pred,gt,channel)

list_of_lists ={'hed_small_ac':hed_small_ac,'hed_small_pr':hed_small_pr,'hed_small_sp':hed_small_sp,'hed_small_se':hed_small_se,'hed_small_fs':hed_small_fs,'hed_ac':hed_ac,'hed_pr':hed_pr,'hed_sp':hed_sp,'hed_se':hed_se,'hed_fs':hed_fs,'cedn_small_ac':cedn_small_ac,'cedn_small_pr':cedn_small_pr,'cedn_small_sp':cedn_small_sp,'cedn_small_se':cedn_small_se,'cedn_small_fs':cedn_small_fs,'cedn_ac':cedn_ac,'cedn_pr':cedn_pr,'cedn_sp':cedn_sp,'cedn_se':cedn_se,'cedn_fs':cedn_fs}

#write CHL metrics to output file
with open('chl_boa.csv', 'w') as f:
    writer = csv.writer(f)
    for key in list_of_lists.keys():
        writer.writerow(key)
        writer.writerow(list_of_lists[key])

#calculate SST best metrics
channel=1
hed_small_ac,hed_small_pr,hed_small_sp,hed_small_se,hed_small_fs=find_best_metrics(hed_small_model_pred,gt,channel)
hed_ac,hed_pr,hed_sp,hed_se,hed_fs=find_best_metrics(hed_model_pred,gt,channel)
cedn_small_ac,cedn_small_pr,cedn_small_sp,cedn_small_se,cedn_small_fs=find_best_metrics(cedn_small_model_pred,gt,channel)
cedn_ac,cedn_pr,cedn_sp,cedn_se,cedn_fs=find_best_metrics(cedn_model_pred,gt,channel)

list_of_lists ={'hed_small_ac':hed_small_ac,'hed_small_pr':hed_small_pr,'hed_small_sp':hed_small_sp,'hed_small_se':hed_small_se,'hed_small_fs':hed_small_fs,'hed_ac':hed_ac,'hed_pr':hed_pr,'hed_sp':hed_sp,'hed_se':hed_se,'hed_fs':hed_fs,'cedn_small_ac':cedn_small_ac,'cedn_small_pr':cedn_small_pr,'cedn_small_sp':cedn_small_sp,'cedn_small_se':cedn_small_se,'cedn_small_fs':cedn_small_fs,'cedn_ac':cedn_ac,'cedn_pr':cedn_pr,'cedn_sp':cedn_sp,'cedn_se':cedn_se,'cedn_fs':cedn_fs}

#write SST metrics to output file
with open('sst_boa.csv', 'w') as f:
    writer = csv.writer(f)
    for key in list_of_lists.keys():
        writer.writerow(key)
        writer.writerow(list_of_lists[key])