
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

from pathlib import Path
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

SCAN_DATA_PATH = '/home/bougoue/Desktop/RockArtObjectDetection/data/Tanum_255_1_part_2'

def ms_output(seconds):
	
	return str(pd.to_timedelta(seconds, unit='s'))

def main():

	model_path = os.path.join(MODELS_PATH, MODEL_NAME)

	# Testing.
	print('\n\nMaking predictions.')

	all_dets = []
	elapsed_times = []

	radnet = load_radnet(os.path.join(model_path, 'config.pickle'))

	img_predict = []
	for img_type in radnet.C.img_types[:]:
		path = Path(SCAN_DATA_PATH) / 'img'
		if 'enhanced_topo' in img_type:
			path = path / 'enhanced_topo_maps'
			if 'grey' in img_type:
				path = path / 'enhanced_topo_map_object_level_grey.png'
			else:
				path = path / 'enhanced_topo_map_object_level.png'

		elif 'blended_map' in img_type:
			path = path / 'blended_maps'
			if 'grey' in img_type:
				path = path / 'blended_map_object_level_grey.png'
			else:
				path = path / 'blended_topo_map_object_level.png'

		elif 'topo' in img_type:
			path = path / 'topo_maps'
			if 'grey' in img_type:
				path = path / 'topo_map_object_level_grey.png'
			else:
				path = path / 'topo_map_object_level.png'

		img_predict.append(cv2.imdecode(
			np.fromfile(str(path), np.uint8),
			cv2.IMREAD_COLOR
		))

	detections = radnet.predict(img_predict)
	predictions = []

	img_viz_path = Path(SCAN_DATA_PATH) / 'img' / 'blended_maps' / 'blended_map_object_level_grey.png'
	img = cv2.imdecode(
		np.fromfile(str(img_viz_path), np.uint8),
		cv2.IMREAD_COLOR
	)

	for d in detections:

		predictions.append({
			'label': d['class'],
			'confidence': float(d['prob']),
			'x1': int(d['x1']),
			'y1': int(d['y1']),
			'x2': int(d['x2']),
			'y2': int(d['y2']),
		})
			
		cv2.rectangle(img, (d['x1'], d['y1']), (d['x2'], d['y2']), (255, 255, 255), 8)
		
		label = '{}: {}'.format(d['class'], int(100*d['prob']))
		retval, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
		label_org = (d['x1'], d['y1']-0)

		cv2.rectangle(img, (label_org[0] - 5, label_org[1]+baseline - 5), (label_org[0]+retval[0] + 5, label_org[1]-retval[1] - 5), (0, 0, 0), 1)
		cv2.rectangle(img, (label_org[0] - 5, label_org[1]+baseline - 5), (label_org[0]+retval[0] + 5, label_org[1]-retval[1] - 5), (255, 255, 255), -1)
		cv2.putText(img, label, label_org, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

	out_path = Path(SCAN_DATA_PATH) / 'img' / 'predictions' / 'all_predictions.png'
	cv2.imwrite(str(out_path), img)

	out_path = Path(SCAN_DATA_PATH) / 'arrays' / 'predictions.json'
	with open(out_path, 'w') as outfile:
		json.dump(predictions, outfile, indent=4)



	img_viz_path = Path(SCAN_DATA_PATH) / 'img' / 'blended_maps' / 'blended_map_object_level_grey.png'
	img = cv2.imdecode(
		np.fromfile(str(img_viz_path), np.uint8),
		cv2.IMREAD_COLOR
	)

	for d in detections:

		if d['class'] == 'boat':
			
			cv2.rectangle(img, (d['x1'], d['y1']), (d['x2'], d['y2']), (28, 26, 228), 8)

	out_path = Path(SCAN_DATA_PATH) / 'img' / 'predictions' / 'boat_predictions.png'
	cv2.imwrite(str(out_path), img)



	img_viz_path = Path(SCAN_DATA_PATH) / 'img' / 'blended_maps' / 'blended_map_object_level_grey.png'
	img = cv2.imdecode(
		np.fromfile(str(img_viz_path), np.uint8),
		cv2.IMREAD_COLOR
	)

	for d in detections:

		if d['class'] == 'human':
			
			cv2.rectangle(img, (d['x1'], d['y1']), (d['x2'], d['y2']), (184, 126, 55), 8)

	out_path = Path(SCAN_DATA_PATH) / 'img' / 'predictions' / 'human_predictions.png'
	cv2.imwrite(str(out_path), img)



	img_viz_path = Path(SCAN_DATA_PATH) / 'img' / 'blended_maps' / 'blended_map_object_level_grey.png'
	img = cv2.imdecode(
		np.fromfile(str(img_viz_path), np.uint8),
		cv2.IMREAD_COLOR
	)

	for d in detections:

		if d['class'] != 'boat' and d['class'] != 'human':
			
			cv2.rectangle(img, (d['x1'], d['y1']), (d['x2'], d['y2']), (0, 127, 255), 8)
			
			label = '{}: {}'.format(d['class'], int(100*d['prob']))
			retval, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
			label_org = (d['x1'], d['y1']-0)

			cv2.rectangle(img, (label_org[0] - 5, label_org[1]+baseline - 5), (label_org[0]+retval[0] + 5, label_org[1]-retval[1] - 5), (0, 0, 0), 1)
			cv2.rectangle(img, (label_org[0] - 5, label_org[1]+baseline - 5), (label_org[0]+retval[0] + 5, label_org[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, label, label_org, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

	out_path = Path(SCAN_DATA_PATH) / 'img' / 'predictions' / 'other_predictions.png'
	cv2.imwrite(str(out_path), img)


if __name__ == '__main__':

	main()