
import os

# Enabled to use the CPU for training.
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import cv2
import json
import shutil
import pickle
import random
import string
import traceback

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from keras.utils import generic_utils
from sklearn.metrics import average_precision_score

from tqdm import tqdm
from datetime import datetime
from faster_rcnn import config
from faster_rcnn.utils import *
from faster_rcnn.rpn import *
from faster_rcnn.losses import *
from faster_rcnn.RADNet import *

MODELS_PATH = 'models'
MODEL_NAME = 'faster_rcnn_resnet50_190902_Monoceros_Mateo'

TEST_ANNOT_PATH = 'data/train.csv'
TEST_DATA_PATH = 'data/train'

def ms_output(seconds):
	
	return str(pd.to_timedelta(seconds, unit='s'))

def calc_class_ap(y_true, y_pred):

	y_true = np.array(y_true)
	y_pred = np.array(y_pred)

	n_gt = np.sum(y_true)

	sorted_idx = np.flip(np.argsort(y_pred))

	tp = 0
	fp = 0
	precision = []
	recall = []

	for idx in sorted_idx:

		if y_true[idx] > 0 and y_pred[idx] > 0.0:
			tp += 1
		elif y_true[idx] == 0 and y_pred[idx] > 0.0:
			fp += 1

		if tp + fp == 0:
			precision.append(0.0)
		else:
			precision.append(tp / (tp + fp))

		if n_gt != 0:
			recall.append(tp / n_gt)
		else:
			recall.append(0.0)


	max_precision = 0.0
	acc_precision = 0.0

	recall_unique, recall_unique_idx = np.unique(recall, return_index=True)

	for i in reversed(recall_unique_idx):

		if precision[i] > max_precision:
			max_precision = precision[i]

		acc_precision += max_precision

	return acc_precision / len(precision)



def main():

	model_path = os.path.join(MODELS_PATH, MODEL_NAME)

	# Testing.
	print('\n\nMaking predictions on TEST data.')

	T = {}
	P = {}
	elapsed_times = []

	radnet = load_radnet(os.path.join(model_path, 'config.pickle'))

	data_test, _, _ = get_data(TEST_ANNOT_PATH, TEST_DATA_PATH, radnet.C.img_types)

	for idx, img in enumerate(data_test[:3]):

		print(img['filepath'] + ' (' + str(idx+1) + '/' + str(len(data_test)) + ')')

		X = get_image(
			img['filepath'],
			radnet.C.img_types,
			random_type=False
		)

		start_time = time.time()

		detections, true, pred = radnet.predict_region_proposals(X, img['bboxes'])

		elapsed_times.append(time.time() - start_time)
		
		for d in detections:
			
			cv2.rectangle(X, (d['x1'], d['y1']), (d['x2'], d['y2']), (255, 255, 255), 8)
			
			label = '{}: {}'.format(d['class'], int(100*d['prob']))
			retval, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
			label_org = (d['x1'], d['y1']-0)

			cv2.rectangle(X, (label_org[0] - 5, label_org[1]+baseline - 5), (label_org[0]+retval[0] + 5, label_org[1]-retval[1] - 5), (0, 0, 0), 1)
			cv2.rectangle(X, (label_org[0] - 5, label_org[1]+baseline - 5), (label_org[0]+retval[0] + 5, label_org[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(X, label, label_org, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

		cv2.imwrite(os.path.join(model_path, os.path.join('test_rpn', img['filepath'].split('/')[-1])), X)

		for key in true.keys():
			if key not in T:
				T[key] = []
				P[key] = []
			T[key].extend(true[key])
			P[key].extend(pred[key])

	'''
	# Calculate mAP.
	all_aps = []
	accuracy = {}
	for key in T.keys():

		ap = calc_class_ap(T[key], P[key])

		print('{} AP: {}'.format(key, ap))
		print('')

		accuracy[key] = ap
		all_aps.append(ap)

	mAP = np.mean(np.array(all_aps))
	accuracy['mAP'] = mAP
	
	with open(os.path.join(model_path, 'test_accuracy.json'), 'w') as outfile:
		json.dump(accuracy, outfile, indent=4)

	print('mAP: ' + str(mAP))
	print('Average prediction time: ' + ms_output(np.mean(elapsed_times)))
	'''


if __name__ == '__main__':

	main()