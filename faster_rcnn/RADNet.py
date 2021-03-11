

import os
import cv2
import sys
import time
import json
import pprint
import random
import pickle

import pandas as pd
import numpy as np

from pathlib import Path
from keras.layers import Input
from keras.models import Model
from tqdm import tqdm

from . import rpn
from faster_rcnn.rpn import *
from faster_rcnn.utils import *


class RADNet():

	"""

		Class for performing detection of rock art.

	"""

	def __init__(self, C, model_rpn, model_detector, preprocess_func):

		self.is_object_threshold = 0.5
		self.bbox_threshold = 0.7
		self.C = C
		self.model_rpn = model_rpn
		self.model_detector = model_detector
		self.preprocess_func = preprocess_func
		self.class_mapping = {v: k for k, v in C.class_mapping.items()}

	# Method to transform the coordinates of the bounding box to its original size
	def get_real_coordinates(self, ratio, x1, y1, x2, y2):

		real_x1 = int(round(x1 // ratio))
		real_y1 = int(round(y1 // ratio))
		real_x2 = int(round(x2 // ratio))
		real_y2 = int(round(y2 // ratio))

		return (real_x1, real_y1, real_x2, real_y2)

	def format_img_size(self, img):

		"""

			Formats the image size based on config.

		"""
		img_min_side = float(self.C.img_size)
		(height,width,_) = img.shape
			
		if width <= height:
			ratio = img_min_side/width
			new_height = int(ratio * height)
			new_width = int(img_min_side)
		else:
			ratio = img_min_side/height
			new_width = int(ratio * width)
			new_height = int(img_min_side)

		img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

		return img, ratio

	def format_img_channels(self, img):

		""" 

			Formats the image channels based on config

		"""
		img = img[:, :, (2, 1, 0)]  # BGR -> RGB
		img = img.astype(np.float32)

		img = np.expand_dims(img, axis=0)
		img = self.preprocess_func(img)

		return img

	def format_img(self, img):

		"""

			Formats an image for model prediction based on config

		"""

		img, ratio = self.format_img_size(img)
		img = self.format_img_channels(img)

		return img, ratio

	def apply_spatial_pyramid_pooling(self, R, feature_map):

		bboxes = {}
		probs = {}

		for jk in range(R.shape[0]//self.C.n_rois + 1):
			ROIs = np.expand_dims(R[self.C.n_rois*jk:self.C.n_rois*(jk+1), :], axis=0)

			if ROIs.shape[1] == 0:
				break

			if jk == R.shape[0]//self.C.n_rois:
				#pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0], self.C.n_rois, curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = self.model_detector.predict([feature_map, ROIs])

			# Calculate bboxes coordinates on resized image
			for ii in range(P_cls.shape[1]):

				# Ignore 'bg' class
				if np.max(P_cls[0, ii, :]) < self.bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]
				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
					tx /= self.C.classifier_regr_std[0]
					ty /= self.C.classifier_regr_std[1]
					tw /= self.C.classifier_regr_std[2]
					th /= self.C.classifier_regr_std[3]
					x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass

				bboxes[cls_name].append([self.C.rpn_stride*x, self.C.rpn_stride*y, self.C.rpn_stride*(x+w), self.C.rpn_stride*(y+h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))
		
		return bboxes, probs

	def final_nms(self, boxes, probs, obj_avg_threshold=0.2, obj_confidence_threshold=0.8, n_obj_avg=5,):
		
		# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
		# if there are no boxes, return an empty list

		# Process explanation:
		#   Step 1: Sort the probs list
		#   Step 2: Find the larget prob 'Last' in the list and save it to the pick list
		#   Step 3: Calculate the IoU with 'Last' box and other boxes in the list. If the IoU is larger than overlap_threshold, delete the box from list
		#   Step 4: Repeat step 2 and step 3 until there is no item in the probs list 
		
		if len(boxes) == 0:
			return []

		# grab the coordinates of the bounding boxes
		x1 = boxes[:, 0]
		y1 = boxes[:, 1]
		x2 = boxes[:, 2]
		y2 = boxes[:, 3]

		np.testing.assert_array_less(x1, x2)
		np.testing.assert_array_less(y1, y2)

		# if the bounding boxes integers, convert them to floats --
		# this is important since we'll be doing a bunch of divisions
		if boxes.dtype.kind == "i":
			boxes = boxes.astype("float")

		# initialize the list of picked indexes	
		pick = []

		# calculate the areas
		area = (x2 - x1) * (y2 - y1)

		# sort the bounding boxes 
		idxs = np.argsort(probs)

		# keep looping while some indexes still remain in the indexes
		# list
		while len(idxs) > 0:
			
			# grab the last index in the indexes list and add the
			# index value to the list of picked indexes
			last = len(idxs) - 1
			i = idxs[last]

			# find the intersection

			xx1_int = np.maximum(x1[i], x1[idxs[:last]])
			yy1_int = np.maximum(y1[i], y1[idxs[:last]])
			xx2_int = np.minimum(x2[i], x2[idxs[:last]])
			yy2_int = np.minimum(y2[i], y2[idxs[:last]])

			ww_int = np.maximum(0, xx2_int - xx1_int)
			hh_int = np.maximum(0, yy2_int - yy1_int)

			area_int = ww_int * hh_int

			# find the union
			area_union = area[i] + area[idxs[:last]] - area_int

			# compute the ratio of overlap
			overlap = area_int/(area_union + 1e-6)

			# delete all indexes from the index list that have.
			
			pick_idx = np.concatenate((np.where(overlap > obj_avg_threshold)[0], [last]))
			
			if probs[idxs[pick_idx]].max() < obj_confidence_threshold:
				conf_idx = idxs[pick_idx][-n_obj_avg:]
			else:
				conf_idx = [_id[0] for _id in np.argwhere(probs[idxs[pick_idx]] > obj_confidence_threshold)]
				conf_idx = idxs[pick_idx][conf_idx]
			#conf_idx = idxs[pick_idx][-n_obj_avg:]

			pick.append(conf_idx)
			idxs = np.delete(idxs, pick_idx)

		new_boxes = []
		new_probs = []
		for p in pick:
			new_boxes.append(np.rint(boxes[p].mean(axis=0)).astype('int'))
			new_probs.append(probs[p].mean())

		return np.array(new_boxes), np.array(new_probs)

		'''
		tmp_boxes = []
		tmp_probs = []
		for p in pick:
			if obj_avg_method == 'mean':
				tmp_boxes.append(boxes[p].mean(axis=0))
				tmp_probs.append(probs[p].mean())
			elif obj_avg_method == 'max':
				tmp_boxes.append([
					boxes[p][:, 0].min(axis=0),
					boxes[p][:, 1].min(axis=0),
					boxes[p][:, 2].max(axis=0),
					boxes[p][:, 3].max(axis=0)
				])
				tmp_probs.append(probs[p].max())

		tmp_boxes = np.array(tmp_boxes)
		tmp_probs = np.array(tmp_probs)

		# Merge new bboxes.

		x1 = tmp_boxes[:, 0]
		y1 = tmp_boxes[:, 1]
		x2 = tmp_boxes[:, 2]
		y2 = tmp_boxes[:, 3]

		np.testing.assert_array_less(x1, x2)
		np.testing.assert_array_less(y1, y2)

		pick = []
		area = (x2 - x1) * (y2 - y1)
		idxs = np.argsort(tmp_probs)

		while len(idxs) > 0:

			last = len(idxs) - 1
			i = idxs[last]

			xx1_int = np.maximum(x1[i], x1[idxs[:last]])
			yy1_int = np.maximum(y1[i], y1[idxs[:last]])
			xx2_int = np.minimum(x2[i], x2[idxs[:last]])
			yy2_int = np.minimum(y2[i], y2[idxs[:last]])

			ww_int = np.maximum(0, xx2_int - xx1_int)
			hh_int = np.maximum(0, yy2_int - yy1_int)

			area_int = ww_int * hh_int
			area_union = area[i] + area[idxs[:last]] - area_int
			overlap = area_int/(area_union + 1e-6)
			
			pick_idx = np.concatenate((np.where(overlap > obj_merge_threshold)[0], [last]))

			pick.append(idxs[pick_idx])
			idxs = np.delete(idxs, pick_idx)

		new_boxes = []
		new_probs = []
		for p in pick:
			new_boxes.append([
				np.rint(tmp_boxes[p][:, 0].min(axis=0)).astype('int'),
				np.rint(tmp_boxes[p][:, 1].min(axis=0)).astype('int'),
				np.rint(tmp_boxes[p][:, 2].max(axis=0)).astype('int'),
				np.rint(tmp_boxes[p][:, 3].max(axis=0)).astype('int')
			])
			new_probs.append(tmp_probs[p].max())

		return np.array(new_boxes), np.array(new_probs)
		'''
	def predict_region_proposals(self, img, gt_bboxes):

		bbox_total = []
		
		if self.C.max_n_tiles_train > 0:

			# Tile
			img_width = img.shape[1]
			img_height = img.shape[0]
			tile_size = self.C.tile_size
			step_size = self.C.tile_overlap

			x_tile_start =  np.arange(0, img_width, step_size)
			x_tile_end = x_tile_start + tile_size	
			mask = np.where(x_tile_end <= img_width)
			x_tile_start = x_tile_start[mask]
			x_tile_end = x_tile_end[mask]
			x_tile_start = np.append(x_tile_start, [max(0, img_width - tile_size)])
			x_tile_end = np.append(x_tile_end, [img_width])
			x_tiles = np.unique([tuple([x_tile_start[i], x_tile_end[i]]) for i in range(x_tile_start.shape[0])], axis=0)

			y_tile_start =  np.arange(0, img_height, step_size)
			y_tile_end = y_tile_start + tile_size
			mask = np.where(y_tile_end <= img_height)
			y_tile_start = y_tile_start[mask]
			y_tile_end = y_tile_end[mask]
			y_tile_start = np.append(y_tile_start, [max(0, img_height - tile_size)])
			y_tile_end = np.append(y_tile_end, [img_height])
			y_tiles = np.unique([tuple([y_tile_start[i], y_tile_end[i]]) for i in range(y_tile_start.shape[0])], axis=0)

			tiles = []
			for y in y_tiles:
				for x in x_tiles:
					tiles.append([x[0], y[0], x[1], y[1]])

			pbar = tqdm(total=len(tiles))
			for tile in tiles:
				
				img_tile = np.copy(img[tile[1]:tile[3], tile[0]:tile[2], :])

				X, ratio = self.format_img(img_tile)

				# Get output layer Y1, Y2 from the RPN and the feature maps F
				# Y1: y_rpn_cls
				# Y2: y_rpn_regr
				[Y1, Y2, F] = self.model_rpn.predict(X)

				import matplotlib
				import matplotlib.pyplot as plt

				t = np.sum(Y1, axis=-1)
				
				for i in range(9):
					fig, ax = plt.subplots()
					im = ax.imshow(Y1[0, :, :, i])
					plt.colorbar(im)
					plt.title(str(i))
					plt.show()

				# Get bboxes by applying NMS 
				# R.shape = (300, 4)
				R = rpn.rpn_to_roi(
					Y1,
					Y2,
					self.C,
					overlap_thresh=0.7
				)

				# convert from (x1,y1,x2,y2) to (x,y,w,h)
				#R[:, 2] -= R[:, 0]
				#R[:, 3] -= R[:, 1]
				R *= 16

				bboxes = R

				for jk in range(bboxes.shape[0]):

					(x1, y1, x2, y2) = bboxes[jk,:]

					# Calculate real coordinates on original image
					(real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)

					bbox_total.append([
						tile[0] + real_x1,
						tile[1] + real_y1,
						tile[0] + real_x2,
						tile[1] + real_y2
					])

				pbar.update(1)

				#break

			pbar.close()

		if self.C.include_full_img:

			X, ratio = self.format_img(img)

			# Get output layer Y1, Y2 from the RPN and the feature maps F
			# Y1: y_rpn_cls
			# Y2: y_rpn_regr
			[Y1, Y2, F] = self.model_rpn.predict(X)

			Y1[0, :, :, 6] = 0.0
			Y1[0, :, :, 7] = 0.0
			Y1[0, :, :, 8] = 0.0

			import matplotlib
			import matplotlib.pyplot as plt

			t = np.sum(Y1, axis=-1)
			
			for i in range(9):
				fig, ax = plt.subplots()
				im = ax.imshow(Y1[0, :, :, i])
				plt.colorbar(im)
				plt.title(str(i))
				plt.show()

			# Get bboxes by applying NMS 
			# R.shape = (300, 4)
			R = rpn.rpn_to_roi(
				Y1,
				Y2,
				self.C,
				overlap_thresh=0.7
			)

			# convert from (x1,y1,x2,y2) to (x,y,w,h)
			R[:, 2] -= R[:, 0]
			R[:, 3] -= R[:, 1]

			#R *= 16

			bboxes = R

			for jk in range(bboxes.shape[0]):

				(x1, y1, x2, y2) = bboxes[jk,:]

				# Calculate real coordinates on original image
				(real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)

				bbox_total.append([
					real_x1,
					real_y1,
					real_x2,
					real_y2
				])

		bbox_total = np.array(bbox_total)

		all_dets = []

		for jk in range(bbox_total.shape[0]):

			(x1, y1, x2, y2) = bbox_total[jk,:]

			all_dets.append({
				'class': 'object',
				'prob': 1.0,
				'x1': x1,
				'y1': y1,
				'x2': x2,
				'y2': y2
			})

		true, pred = self.get_map(all_dets, gt_bboxes, ratio)

		return all_dets, true, pred

	def predict_from_path(self, img_path):

		images = []

		if self.C.use_img_type:
			for img_type in self.C.img_types:
				images.append(get_image(
					img_path,
					[img_type],
					random_type=False
				))
		else:
			images.append(get_image(
				img_path,
				self.C.img_types,
				random_type=False
			))

		return self.predict(images)

	def predict(self, images):

		all_img_all_bbox = {}
		all_img_all_probs = {}
		for img in images:

			bbox_total = {}
			probs_total = {}
			
			if self.C.max_n_tiles_train > 0:

				# Tile
				img_width = img.shape[1]
				img_height = img.shape[0]
				tile_size = self.C.tile_size
				step_size = self.C.tile_overlap

				x_tile_start =  np.arange(0, img_width, step_size)
				x_tile_end = x_tile_start + tile_size	
				mask = np.where(x_tile_end <= img_width)
				x_tile_start = x_tile_start[mask]
				x_tile_end = x_tile_end[mask]
				x_tile_start = np.append(x_tile_start, [max(0, img_width - tile_size)])
				x_tile_end = np.append(x_tile_end, [img_width])
				x_tiles = np.unique([tuple([x_tile_start[i], x_tile_end[i]]) for i in range(x_tile_start.shape[0])], axis=0)

				y_tile_start =  np.arange(0, img_height, step_size)
				y_tile_end = y_tile_start + tile_size
				mask = np.where(y_tile_end <= img_height)
				y_tile_start = y_tile_start[mask]
				y_tile_end = y_tile_end[mask]
				y_tile_start = np.append(y_tile_start, [max(0, img_height - tile_size)])
				y_tile_end = np.append(y_tile_end, [img_height])
				y_tiles = np.unique([tuple([y_tile_start[i], y_tile_end[i]]) for i in range(y_tile_start.shape[0])], axis=0)

				tiles = []
				for y in y_tiles:
					for x in x_tiles:
						tiles.append([x[0], y[0], x[1], y[1]])

				pbar = tqdm(total=len(tiles))
				for tile in tiles:
					
					img_tile = np.copy(img[tile[1]:tile[3], tile[0]:tile[2], :])

					X, ratio = self.format_img(img_tile)

					# Get output layer Y1, Y2 from the RPN and the feature maps F
					# Y1: y_rpn_cls
					# Y2: y_rpn_regr
					[Y1, Y2, F] = self.model_rpn.predict(X)

					# Get bboxes by applying NMS 
					# R.shape = (300, 4)
					R = rpn.rpn_to_roi(
						Y1,
						Y2,
						self.C,
						overlap_thresh=0.7
					)

					# convert from (x1,y1,x2,y2) to (x,y,w,h)
					R[:, 2] -= R[:, 0]
					R[:, 3] -= R[:, 1]

					# Apply the spatial pyramid pooling to the proposed regions
					bboxes, probs = self.apply_spatial_pyramid_pooling(R, F)

					for key in bboxes:

						bbox = np.array(bboxes[key])

						new_boxes, new_probs = rpn.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
						for jk in range(new_boxes.shape[0]):

							(x1, y1, x2, y2) = new_boxes[jk,:]

							# Calculate real coordinates on original image
							(real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)

							if key in bbox_total:
								bbox_total[key].append([
									tile[0] + real_x1,
									tile[1] + real_y1,
									tile[0] + real_x2,
									tile[1] + real_y2
								])
							else:
								bbox_total[key] = [[
									tile[0] + real_x1,
									tile[1] + real_y1,
									tile[0] + real_x2,
									tile[1] + real_y2
								]]

							if key in probs_total:
								probs_total[key].append(new_probs[jk])
							else:
								probs_total[key] = [new_probs[jk]]

					pbar.update(1)

				pbar.close()

			if self.C.include_full_img:

				X, ratio = self.format_img(img)

				#X = np.transpose(X, (0, 2, 3, 1))

				# Get output layer Y1, Y2 from the RPN and the feature maps F
				# Y1: y_rpn_cls
				# Y2: y_rpn_regr
				[Y1, Y2, F] = self.model_rpn.predict(X)

				# Get bboxes by applying NMS 
				# R.shape = (300, 4)
				R = rpn.rpn_to_roi(
					Y1,
					Y2,
					self.C,
					overlap_thresh=0.7
				)

				# convert from (x1,y1,x2,y2) to (x,y,w,h)
				R[:, 2] -= R[:, 0]
				R[:, 3] -= R[:, 1]

				# Apply the spatial pyramid pooling to the proposed regions
				bboxes, probs = self.apply_spatial_pyramid_pooling(R, F)

				all_dets = []

				for key in bboxes:

					bbox = np.array(bboxes[key])

					new_boxes, new_probs = rpn.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
					for jk in range(new_boxes.shape[0]):

						(x1, y1, x2, y2) = new_boxes[jk,:]

						# Calculate real coordinates on original image
						(real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)

						if key in bbox_total:
							bbox_total[key].append([
								real_x1,
								real_y1,
								real_x2,
								real_y2
							])
						else:
							bbox_total[key] = [[
								real_x1,
								real_y1,
								real_x2,
								real_y2
							]]

						if key in probs_total:
							probs_total[key].append(new_probs[jk])
						else:
							probs_total[key] = [new_probs[jk]]

			
			all_dets = []

			for key in bbox_total:
				
				new_boxes, new_probs = self.final_nms(
					np.array(bbox_total[key]),
					np.array(probs_total[key]),
					obj_avg_threshold=0.2,
					obj_confidence_threshold=0.8,
					n_obj_avg=5,
				)
				
				#new_boxes, new_probs = np.array(bbox_total[key]), np.array(probs_total[key])
				for jk in range(new_boxes.shape[0]):

					(x1, y1, x2, y2) = new_boxes[jk,:]

					if key in all_img_all_bbox:
						all_img_all_bbox[key].append([x1, y1, x2, y2])
					else:
						all_img_all_bbox[key] = [[x1, y1, x2, y2]]

					if key in all_img_all_probs:
						all_img_all_probs[key].append(new_probs[jk])
					else:
						all_img_all_probs[key] = [new_probs[jk]]

		all_img_all_dets = []
		for key in all_img_all_bbox:
			
			new_boxes, new_probs = rpn.non_max_suppression_fast(
				np.array(all_img_all_bbox[key]),
				np.array(all_img_all_probs[key]),
				overlap_thresh=0.4
			)
			#new_boxes, new_probs = np.array(bbox_total[key]), np.array(probs_total[key])

			for jk in range(new_boxes.shape[0]):

				(x1, y1, x2, y2) = new_boxes[jk,:]

				all_img_all_dets.append({
					'class': key,
					'prob': new_probs[jk],
					'x1': x1,
					'y1': y1,
					'x2': x2,
					'y2': y2
				})

		return all_img_all_dets


def load_radnet(config_path):


	C = pickle.load(open(config_path, 'rb'))

	# Load base model.
	if C.network == 'vgg16':
		from faster_rcnn.base_models import vgg16 as base_model
	elif C.network == 'resnet50':
		from faster_rcnn.base_models import resnet50 as base_model
	else:
		print('Not a valid base model!')
		sys.exit(1)	

	if C.network == 'vgg16':
		n_features = 512
	elif C.network == 'resnet50':
		n_features = 1024

	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, n_features)

	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.n_rois, 4))
	feature_map_input = Input(shape=input_shape_features)

	# Define base network with shared layers.
	base_net_output_layer = base_model.nn_base(img_input, trainable=True, weights=None)

	# Load RPN Network.
	n_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn_layers = rpn_layer(base_net_output_layer, n_anchors)
	model_rpn = Model(img_input, rpn_layers)
	model_rpn.load_weights(Path(C.weights_path.replace('\\', '/')), by_name=True)
	model_rpn.compile(optimizer='sgd', loss='mse')

	#for i, layer in enumerate(model_rpn.layers):
	#	print(i, layer.name, layer.output_shape, layer.trainable)
	
	# Load detector.
	detector_layers = base_model.classifier_layer(
		feature_map_input,
		roi_input,
		C.n_rois,
		nb_classes=len(C.class_mapping)
	)
	#model_detector_only = Model([feature_map_input, roi_input], detector_layers)
	model_detector = Model([feature_map_input, roi_input], detector_layers)
	model_detector.load_weights(Path(C.weights_path.replace('\\', '/')), by_name=True)
	model_detector.compile(optimizer='sgd', loss='mse')

	#for i, layer in enumerate(model_detector.layers):
	#	print(i, layer.name, layer.output_shape, layer.trainable)

	return RADNet(C, model_rpn, model_detector, base_model.preprocess)