from __future__ import generator_stop

import os

import sys
import cv2
import json
import shutil
import pickle
import random
import string
import traceback
import random
import urllib.request

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from tqdm import tqdm
from datetime import datetime
from faster_rcnn import config
from faster_rcnn.utils import *


SEED = 64

TRAIN_ANNOT_PATH = 'data/train.csv'
TRAIN_DATA_PATH = 'data/train'

def main():

		# Set seed.
	np.random.seed(SEED)

	# Read config.
	C = config.Config()

	# Read training data.
	data_train, class_count, _ = get_data(TRAIN_ANNOT_PATH, TRAIN_DATA_PATH, C.img_types)

	# Load base model.
	if C.network == 'vgg16':
		from faster_rcnn.base_models import vgg16 as base_model
	elif C.network == 'resnet50':
		from faster_rcnn.base_models import resnet50 as base_model
	else:
		print('Not a valid base model!')
		sys.exit(1)

	# Counting epoch length.
	'''
	if True:

		print('Computing Epoch Length and Anchors Ratios/Sizes.')

		df = pd.DataFrame()

		data_train_gen = get_tile_generator(
			data_train,
			C,
			base_model.get_img_output_length,
			class_count,
			base_model.preprocess,
			train_mode=False,
			verbose=False
		)

		anchors_summary = {}
		for size in C.anchor_box_scales:
			anchors_summary[str(size)] = {}
			for ratio in C.anchor_box_ratios:
				anchors_summary[str(size)][str(ratio)] = 0

		n_objects = []
		epoch_length_counter = 0
		while True:

			try:
				img, Y, img_data, img_debug, best_anchor_for_bbox, debug_n_pos = next(data_train_gen)
				n_objects.append(len(img_data['bboxes']))

				for i in range(len(img_data['bboxes'])):

					gt_x1 = int(img_data['bboxes'][i]['x1']*(img.shape[2]/img_data['width']))
					gt_x2 = int(img_data['bboxes'][i]['x2']*(img.shape[2]/img_data['width']))
					gt_y1 = int(img_data['bboxes'][i]['y1']*(img.shape[1]/img_data['height']))
					gt_y2 = int(img_data['bboxes'][i]['y2']*(img.shape[1]/img_data['height']))

					ar_x = (gt_x2 - gt_x1)
					ar_y = (gt_y2 - gt_y1)

					df = df.append(
						{
							'new_width': img.shape[1],
							'new_height': img.shape[2],
							'bbox_width': gt_x2 - gt_x1,
							'bbox_height': gt_y2 - gt_y1,
							'bbox_aspect_ratio': max(ar_x, ar_y) / min(ar_x, ar_y)
						},
						ignore_index=True
					)



				_cls = Y[0][0]
				_regr = Y[1][0]
				pos_cls = np.where(_cls==1)
				pos_regr = np.where(_regr==1)

				for i in range(debug_n_pos):

					idx = pos_regr[2][i*4]/4
					try:
						anchor_size = C.anchor_box_scales[int(idx/len(C.anchor_box_ratios))]
						anchor_ratio = C.anchor_box_ratios[int(idx%len(C.anchor_box_ratios))]
					except:
						print(debug_n_pos)
						print(i)
						print(idx)
						print(pos_regr[2])

					anchors_summary[str(anchor_size)][str(anchor_ratio)] += 1

				epoch_length_counter += 1

				#if epoch_length_counter > 20:
				#	break

			except RuntimeError:
				break	
			except StopIteration:
				break	

		print('Epoch Length: ' + str(epoch_length_counter))
		print('Min Nr of Objects: ' + str(min(n_objects)))
		print('Average Nr of Objects: ' + str(np.average(n_objects)))
		print('Max Nr of Objects: ' + str(max(n_objects)))
		print(anchors_summary)
		print('\n')


	# Count Base Sizes
	base_sizes = [32, 64, 96, 128, 196, 212, 256, 512, 768, 1024]
	base_sizes_results = {bs: 0 for bs in base_sizes}
	base_sizes_results['rest'] = 0

	for index, row in df.iterrows():
		width = row['bbox_width']
		height = row['bbox_height']

		done = False
		for bs in base_sizes:
			if width <= bs and height <= bs:
				base_sizes_results[bs] += 1
				done = True

		if done == False:
			base_sizes_results['rest'] += 1

	plt.figure(figsize=(12,8))
	plt.bar(
		range(len(base_sizes_results)),
		[base_sizes_results[key] for key in base_sizes_results],
		align='center'
	)
	plt.xticks(range(len(base_sizes_results)), base_sizes_results.keys())
	plt.show()

	plt.figure(figsize=(12,8))
	sns.distplot(df['bbox_aspect_ratio'], bins=20)
	plt.show()

	# Clustering Scales.
	X = df.as_matrix(columns=['bbox_width', 'bbox_height'])

	K = KMeans(3, random_state=SEED)
	labels = K.fit(X)
	plt.figure(figsize=(12,8))
	plt.scatter(X[:, 0], X[:, 1], c=labels.labels_, s=50, alpha=0.5, cmap='viridis')
	plt.show()
	'''
	# Test generator.
	
	print('Testing generator for training data.')

	data_train_gen = get_tile_generator(
		data_train,
		C,
		base_model.get_img_output_length,
		class_count,
		base_model.preprocess,
		train_mode=True,
		verbose=True
	)

	continue_test = 'Y'

	while continue_test.upper() == 'Y':

		img, Y, img_data, img_debug, best_anchor_for_bbox, debug_n_pos = next(data_train_gen)

		print('Image: ' + img_data['filepath'])
		print('Original image: height=%d width=%d'%(img_data['height'], img_data['width']))
		print('Resized image:  height=%d width=%d C.img_size=%d'%(img.shape[1], img.shape[2], C.img_size))
		print('Feature Map Size: height=%d width=%d C.rpn_stride=%d'%(Y[0].shape[1], Y[0].shape[2], C.rpn_stride))
		print('Nr of GT Bounding Boxes: ' + str(len(img_data['bboxes'])))
		print('Shape of y_rpn_cls {}'.format(Y[0].shape))
		print('Shape of y_rpn_regr {}'.format(Y[1].shape))
		print('Number of positive anchors for this image: ' + str(debug_n_pos))

		img_debug_gray = img_debug.copy()
		img_debug = cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB)
		img_debug_gray = cv2.cvtColor(img_debug_gray, cv2.COLOR_BGR2GRAY)
		img_debug_gray = cv2.cvtColor(img_debug_gray, cv2.COLOR_GRAY2RGB)
		img_debug_gray_anchors = np.copy(img_debug_gray)

		colormap = [
			((166,206,227), (31,120,180)), # Light Blue, Blue
			((178,223,138), (51,160,44)), # Light Green, Green
			((251,154,153), (227,26,28)), # Light Red, Red
			((253,191,111), (255,127,0)), # Light Orange, Orange
			((202,178,214), (106,61,154)), # Light Purple, Purple
		]

		cc = 0
		#for i in np.random.choice(len(img_data['bboxes']), min(5, len(img_data['bboxes'])), replace=False):
		for i in range(len(img_data['bboxes'])):

			gt_label = img_data['bboxes'][i]['class']
			gt_x1 = int(img_data['bboxes'][i]['x1']*(img.shape[2]/img_data['width']))
			gt_x2 = int(img_data['bboxes'][i]['x2']*(img.shape[2]/img_data['width']))
			gt_y1 = int(img_data['bboxes'][i]['y1']*(img.shape[1]/img_data['height']))
			gt_y2 = int(img_data['bboxes'][i]['y2']*(img.shape[1]/img_data['height']))

			cv2.putText(
				img_debug_gray,
				gt_label,
				(gt_x1, gt_y1-10),
				cv2.FONT_HERSHEY_DUPLEX,
				0.7,
				colormap[cc%len(colormap)][1],
				1
			)
			cv2.rectangle(
				img_debug_gray,
				(gt_x1, gt_y1),
				(gt_x2, gt_y2),
				colormap[cc%len(colormap)][1],
				3
			)

			center = (best_anchor_for_bbox[i, 1]*C.rpn_stride, best_anchor_for_bbox[i, 0]*C.rpn_stride)
			anchor_size = C.anchor_box_scales[best_anchor_for_bbox[i, 3]]
			anchor_ratio = C.anchor_box_ratios[best_anchor_for_bbox[i, 2]]
			anchor_width = anchor_size*anchor_ratio[0]
			anchor_height = anchor_size*anchor_ratio[1]

			cv2.circle(img_debug_gray, center, 3, colormap[cc%len(colormap)][0], -1)

			cv2.rectangle(
				img_debug_gray,
				(center[0]-int(anchor_width/2), center[1]-int(anchor_height/2)),
				(center[0]+int(anchor_width/2), center[1]+int(anchor_height/2)),
				colormap[cc%len(colormap)][0],
				3
			)

			overlay = img_debug.copy()
			cv2.rectangle(
				overlay,
				(gt_x1, gt_y1),
				(gt_x2, gt_y2),
				(227,26,28),
				3
			)
			alpha = 0.6
			img_debug = cv2.addWeighted(overlay, alpha, img_debug, 1 - alpha, 0)

			cc += 1

		_cls = Y[0][0]
		_regr = Y[1][0]
		pos_cls = np.where(_cls==1)
		pos_regr = np.where(_regr==1)

		for i in range(debug_n_pos):

			color = colormap[i%len(colormap)][0]

			idx = pos_regr[2][i*4]/4
			anchor_size = C.anchor_box_scales[int(idx/len(C.anchor_box_ratios))]
			anchor_ratio = C.anchor_box_ratios[int(idx%len(C.anchor_box_ratios))]

			center = (pos_regr[1][i*4]*C.rpn_stride, pos_regr[0][i*4]*C.rpn_stride)
			anchor_width = anchor_size*anchor_ratio[0]
			anchor_height = anchor_size*anchor_ratio[1]

			cv2.circle(img_debug_gray_anchors, center, 3, color, -1)
			cv2.rectangle(
				img_debug_gray_anchors,
				(center[0]-int(anchor_width/2), center[1]-int(anchor_height/2)),
				(center[0]+int(anchor_width/2), center[1]+int(anchor_height/2)),
				color,
				2
			)

		plt.figure(figsize=(8,8))
		plt.imshow(img_debug_gray)
		plt.figure(figsize=(8,8))
		plt.imshow(img_debug_gray_anchors)
		plt.figure(figsize=(8,8))
		plt.imshow(img_debug)
		plt.show()

		cv2.imwrite('test.png', cv2.cvtColor(img_debug, cv2.COLOR_RGB2BGR))

		continue_test = input('Do you want to continue generator test round? Y or N?')

		print('')




if __name__ == '__main__':

	main()
