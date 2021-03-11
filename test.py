
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
MODEL_NAME = 'faster_rcnn_resnet50_raod_base'

TEST_ANNOT_PATH = 'data/test.csv'
TEST_DATA_PATH = 'data/test'

GT_IOU_THRESHOLD = 0.5

def ms_output(seconds):
	
	return str(pd.to_timedelta(seconds, unit='s'))

def get_objects(pred, gt, treshold):

	T = {}
	P = {}

	for bbox in gt:
		bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:

		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		pred_x1 = pred_box['x1']
		pred_y1 = pred_box['y1']
		pred_x2 = pred_box['x2']
		pred_y2 = pred_box['y2']
		pred_prob = pred_box['prob']

		if pred_class not in P:
			P[pred_class] = []
			T[pred_class] = []

		P[pred_class].append(pred_prob)
		found_match = False
		
		for gt_box in gt:
			gt_class = gt_box['class']
			gt_x1 = gt_box['x1']
			gt_y1 = gt_box['y1']
			gt_x2 = gt_box['x2']
			gt_y2 = gt_box['y2']
			gt_seen = gt_box['bbox_matched']

			if gt_class != pred_class:
				continue

			if gt_seen:
				continue

			iou_map = iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))

			if iou_map >= treshold:

				found_match = True
				gt_box['bbox_matched'] = True
				break

			else:
				continue

		T[pred_class].append(int(found_match))

	for gt_box in gt:

		if not gt_box['bbox_matched']:# and not gt_box['difficult']:

			if gt_box['class'] not in P:

				P[gt_box['class']] = []
				T[gt_box['class']] = []

			T[gt_box['class']].append(1)
			P[gt_box['class']].append(0)

	return T, P


# VOC Challenge Metric
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

	precision = np.array(precision)
	recall = np.array(recall)

	max_precision = 0.0
	interpolated_precision = []
	interpolated_recall = []
	recall_unique, recall_unique_idx = np.unique(recall, return_index=True)

	for i in reversed(range(len(recall))):

		if precision[i] > max_precision:
			max_precision = precision[i]

		interpolated_recall.append(recall[i])
		interpolated_precision.append(max_precision)

	interpolated_precision.reverse()
	interpolated_recall.reverse()

	ap = 0
	for i in range(len(interpolated_precision)-1):
		ap += interpolated_precision[i+1] * (interpolated_recall[i+1]-interpolated_recall[i])

	return ap, precision, recall, interpolated_precision, interpolated_recall



def main():

	model_path = os.path.join(MODELS_PATH, MODEL_NAME)

	# Testing.
	print('\n\nMaking predictions on TEST data.')

	all_dets = []
	all_gt = []
	elapsed_times = []

	radnet = load_radnet(os.path.join(model_path, 'config.pickle'))

	data_test, _, _ = get_data(TEST_ANNOT_PATH, TEST_DATA_PATH, radnet.C.img_types)

	for idx, img_meta in enumerate(data_test[:]):

		print(img_meta['filepath'] + ' (' + str(idx+1) + '/' + str(len(data_test)) + ')')

		img = get_image(
			img_meta['filepath'],
			['blended_grey'],
			random_type=False
		)

		start_time = time.time()

		detections = radnet.predict_from_path(img_meta['filepath'])
		elapsed_times.append(time.time() - start_time)
		
		for d in detections:
			
			cv2.rectangle(img, (d['x1'], d['y1']), (d['x2'], d['y2']), (255, 255, 255), 8)
			
			label = '{}: {}'.format(d['class'], int(100*d['prob']))
			retval, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
			label_org = (d['x1'], d['y1']-0)

			cv2.rectangle(img, (label_org[0] - 5, label_org[1]+baseline - 5), (label_org[0]+retval[0] + 5, label_org[1]-retval[1] - 5), (0, 0, 0), 1)
			cv2.rectangle(img, (label_org[0] - 5, label_org[1]+baseline - 5), (label_org[0]+retval[0] + 5, label_org[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, label, label_org, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

		cv2.imwrite(os.path.join(model_path, os.path.join('test', img_meta['filepath'].split('/')[-1])), img)

		all_dets.extend(detections)
		all_gt.extend(img_meta['bboxes'])

	# Compute mAP PASCAL VOC metric
	# https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/
	# https://github.com/rafaelpadilla/Object-Detection-Metrics#precision-x-recall-curve
	T, P = get_objects(all_dets, all_gt, GT_IOU_THRESHOLD)

	all_aps = []
	accuracy = {}
	plt.figure(figsize=(12,12))
	ax = plt.gca()
	for key in sorted(T.keys()):

		ap, precision, recall, interpolated_precision, interpolated_recall = calc_class_ap(T[key], P[key])

		color = next(ax._get_lines.prop_cycler)['color']
		plt.plot(recall, precision, linestyle='-', color=color, label=key + ': ' + '{0:.2f}'.format(100*ap) + ' %')
		plt.plot(interpolated_recall, interpolated_precision, linestyle='--', color=color)

		print('{} AP: {}'.format(key, ap))
		print('')

		accuracy[key] = ap
		all_aps.append(ap)

	mAP = np.mean(np.array(all_aps))
	accuracy['mAP'] = mAP

	plt.ylabel('Precision (TP / TP + FP)')
	plt.xlabel('Recall (TP / TP + FN)')
	plt.ylim(0.0, 1.0)
	plt.xlim(0.0, 1.0)
	plt.title('mAP: ' + '{0:.2f}'.format(100*mAP) + ' %')
	plt.legend()
	plt.savefig(os.path.join(model_path, 'viz/precision_recall.png'))
	
	with open(os.path.join(model_path, 'test_accuracy.json'), 'w') as outfile:
		json.dump(accuracy, outfile, indent=4)

	print('mAP: ' + str(mAP))
	print('Average prediction time: ' + ms_output(np.mean(elapsed_times)))

if __name__ == '__main__':

	main()