
import os
import sys
import cv2
import time
import math
import copy
import pprint
import random
import itertools
import traceback

import pandas as pd
import numpy as np

from tqdm import tqdm
from . import augmentation

class SampleSelector:
	def __init__(self, class_count):
		# ignore classes that have zero samples
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	def skip_image_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				break

		if class_in_img:
			return False
		else:
			return True

	def skip_tile_for_balanced_class(self, img_data):

		class_in_img = False

		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False
		else:
			return True

def ms_output(seconds):
	
	return str(pd.to_timedelta(seconds, unit='s'))

def get_new_img_size(width, height, img_min_side=300):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height

def union(au, bu, area_intersection):

	area_a = (au[2] - au[0]) * (au[3] - au[1])
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
	area_union = area_a + area_b - area_intersection

	return area_union


def intersection(ai, bi):

	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y

	if w < 0 or h < 0:
		return 0

	return w*h


def iou(a, b):

	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)
	area_u = union(a, b, area_i)

	return float(area_i) / float(area_u + 1e-6)

def get_image(img_path, types, random_type=False):

	img_type = types[0]

	if random_type:

		first_prob = 0.3
		if len(types) <= 3:
			first_prob = 0.5

		probs = [first_prob] + [(1.0-first_prob)/(len(types)-1) for i in range(len(types)-1)]
		img_type = np.random.choice(types, 1, p=probs)[0]

	img_path = img_path.split('/')
	img_path.insert(1, img_type)
	img_path = os.path.join(*img_path)
	img = cv2.imdecode(
		np.fromfile(img_path, np.uint8),
		cv2.IMREAD_COLOR
	)

	return img

def get_data(annot_path, data_path, img_types):

	'''

		Parse data from annotation file.

		Args:
			annot_path: annotation file path.
			data_path: data folder.

		Returns:
			data: list(filepath, width, height, depth, list(bboxes))
			class_count: dict{key:class_name, value:count_num}
			class_mapping: dict{key:class_name, value: idx}

	'''

	print('\nReading: ' + annot_path)

	time_start = time.time()

	df = pd.read_csv(annot_path)

	all_imgs = {}
	class_count = {}
	class_mapping = {}
	for i in tqdm(range(df.shape[0])):

		img_name = df.loc[i, 'img_path']
		class_name = df.loc[i, 'label']
		x1 = df.loc[i, 'xmin']
		y1 = df.loc[i, 'ymin']
		x2 = df.loc[i, 'xmax']
		y2 = df.loc[i, 'ymax']

		if class_name not in class_count:
			class_count[class_name] = 1
		else:
			class_count[class_name] += 1

		if class_name not in class_mapping:
			class_mapping[class_name] = len(class_mapping)

		if img_name not in all_imgs:

			all_imgs[img_name] = {}
			#filepath = os.path.join(data_path, img_name)
			filepath = data_path + '/' + img_name

			img = get_image(filepath, img_types, random_type=False)

			(rows, cols, channels) = img.shape

			all_imgs[img_name]['filepath'] = filepath
			all_imgs[img_name]['width'] = cols
			all_imgs[img_name]['height'] = rows
			all_imgs[img_name]['depth'] = channels
			all_imgs[img_name]['bboxes'] = []

		all_imgs[img_name]['bboxes'].append({
			'class': class_name,
			'x1': int(x1),
			'y1': int(y1),
			'x2': int(x2),
			'y2': int(y2)
		})

	data = [all_imgs[key] for key in all_imgs]

	if 'bg' not in class_count:
		class_count['bg'] = 0
		class_mapping['bg'] = len(class_mapping)

	time_end = time.time()
	print('Execution Time: ' + ms_output(time_end - time_start))

	print('Nr of images: ' + str(len(data)))

	print('Bounding boxes per class:')
	pprint.pprint(class_count)

	print('Class Mapping:')
	print(class_mapping)

	print('')

	return data, class_count, class_mapping

def get_generator(data, C, get_feat_map_size, preprocess_func, train_mode=True, verbose=False):


	'''

		Yield the ground-truth anchors as Y-labels.

		Args:
			data: list(filepath, width, height, depth, list(bboxes))
			C: config,
			get_feat_map_size: function to calculate final layer's feature map 
				(of base model) size according to input image size.
			train_mode: If true -> do augmentation.

		Returns:
			x_img: image data after resized and scaling (smallest size = 300px)
			Y: [y_rpn_cls, y_rpn_regr]
			img_data_aug: augmented image data (original image with augmentation)
			debug_img: show image for debug
			num_pos: show number of positive anchors for debug

	'''

	while True:

		if train_mode:
			np.random.shuffle(data)

		for img_data in data:

			#img = cv2.imread(img_data['filepath'])
			img = get_image(img_data['filepath'], C.img_types, random_type=C.use_img_type)

			if train_mode:
				img_data_aug, img = augmentation.augment(img_data, img, C, augment=True, verbose=verbose)
			else:
				img_data_aug, img = augmentation.augment(img_data, img, C, augment=False, verbose=verbose)	

			width = img_data_aug['width']
			height = img_data_aug['height']
			rows = img.shape[0]
			cols = img.shape[1]

			assert cols == width
			assert rows == height

			# Get new image dimensions.
			width_resized, height_resized = get_new_img_size(width, height, C.img_size)

			# Resize image.
			img = cv2.resize(img, (width_resized, height_resized), interpolation=cv2.INTER_CUBIC)

			img_debug = img.copy()

			try:
				y_rpn_cls, y_rpn_regr, best_anchor_for_bbox, n_pos = calc_region_props(
					C,
					img_data_aug,
					width,
					height,
					width_resized,
					height_resized,
					get_feat_map_size,
					verbose=verbose
				)
			except Exception as e:
				print(e)
				continue

			# Zero-center by mean pixel and preprocess image.
			img = img[:, :, (2, 1, 0)] # BGR -> RGB
			img = img.astype(np.float32)

			img = np.expand_dims(img, axis=0)
			img = preprocess_func(img)

			# Not sure what is happening here?
			y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

			y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
			y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

			yield np.copy(img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, img_debug, best_anchor_for_bbox, n_pos
			
		if train_mode == False:
			raise StopIteration


def get_tile_generator(data, C, get_feat_map_size, class_count, preprocess_func, train_mode=True, verbose=False):

	'''

		Yield the ground-truth anchors as Y-labels.

		Args:
			data: list(filepath, width, height, depth, list(bboxes))
			C: config,
			get_feat_map_size: function to calculate final layer's feature map 
				(of base model) size according to input image size.
			train_mode: If true -> do augmentation. If false -> take a few more tiles for validation.

		Returns:
			x_img: image data after resized and scaling (smallest size = 300px)
			Y: [y_rpn_cls, y_rpn_regr]
			img_data_aug: augmented image data (original image with augmentation)
			debug_img: show image for debug
			num_pos: show number of positive anchors for debug

	'''

	sample_selector = SampleSelector(class_count)

	while True:

		if train_mode:
			np.random.shuffle(data)

		for img_data in data:

			if train_mode and C.balanced_classes and sample_selector.skip_image_for_balanced_class(img_data):
				continue

			img = get_image(img_data['filepath'], C.img_types, random_type=False)

			# Tile
			img_width = img_data['width']
			img_height = img_data['height']
			tile_size = C.tile_size
			step_size = C.tile_overlap

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

			if len(tiles) == 0:
				print('\nNo tiles in: ' + img_data['filepath'] + '\n')
				continue

			tile_counter = 0
			all_tile_indices = np.arange(0, len(tiles))
			#idx = -1 #REMOVE
			if train_mode:
				n_tiles = min(len(tiles), C.max_n_tiles_train)
			else:
				n_tiles = min(len(tiles), C.max_n_tiles_val)

			while tile_counter < n_tiles and len(all_tile_indices) > 0:

				# IF TRAINING MODE NO RANDOM TYPE?
				img = get_image(img_data['filepath'], C.img_types, random_type=C.use_img_type)

				idx = np.random.randint(0, len(all_tile_indices))
				#idx += 1 #REMOVE
				tile_idx = all_tile_indices[idx]
				all_tile_indices = np.delete(all_tile_indices, idx)

				tile = tiles[tile_idx]
				
				img_tile = np.copy(img[tile[1]:tile[3], tile[0]:tile[2], :])

				img_data_tile = copy.deepcopy(img_data)
				tile_bboxes = img_data_tile['bboxes']

				bboxes_arr = np.array([[bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']] for bbox in tile_bboxes])
				bboxes_arr, bboxes_mask = augmentation.clip_box(bboxes_arr, [tile[0], tile[1], tile[2], tile[3]], C.tile_bbox_clip_threshold)
				tile_bboxes = [tile_bboxes[i] for i in range(bboxes_mask.shape[0]) if bboxes_mask[i] == 1]

				# If no bboxes in tile.
				if len(tile_bboxes) == 0:
					continue

				for i in range(bboxes_arr.shape[0]):
					tile_bboxes[i]['x1'] = int(bboxes_arr[i, 0]-tile[0])
					tile_bboxes[i]['y1'] = int(bboxes_arr[i, 1]-tile[1])
					tile_bboxes[i]['x2'] = int(math.ceil(bboxes_arr[i, 2]-tile[0]))
					tile_bboxes[i]['y2'] = int(math.ceil(bboxes_arr[i, 3]-tile[1]))

				img_data_tile['width'] = img_tile.shape[1]
				img_data_tile['height'] = img_tile.shape[0]
				img_data_tile['bboxes'] = tile_bboxes

				if train_mode and C.balanced_classes and sample_selector.skip_tile_for_balanced_class(img_data_tile):
					continue

				if train_mode:
					img_data_tile, img_tile = augmentation.augment(img_data_tile, img_tile, C, augment=True, verbose=verbose)
				else:
					img_data_tile, img_tile = augmentation.augment(img_data_tile, img_tile, C, augment=False, verbose=verbose)

				width = img_data_tile['width']
				height = img_data_tile['height']
				rows = img_tile.shape[0]
				cols = img_tile.shape[1]

				assert cols == width
				assert rows == height

				# Get new image dimensions.
				width_resized, height_resized = get_new_img_size(width, height, C.img_size)

				# Resize image.
				img_tile = cv2.resize(
					img_tile,
					(width_resized, height_resized), 
					interpolation=cv2.INTER_CUBIC
				)

				img_debug = img_tile.copy()

				try:
					y_rpn_cls, y_rpn_regr, best_anchor_for_bbox, n_pos = calc_region_props(
						C,
						img_data_tile,
						width,
						height,
						width_resized,
						height_resized,
						get_feat_map_size,
						verbose=verbose
					)
				except Exception as e:
					print('')
					traceback.print_exc()
					print('')
					continue

				# Zero-center by mean pixel and preprocess image.
				img_tile = img_tile[:, :, (2, 1, 0)] # BGR -> RGB
				img_tile = img_tile.astype(np.float32)

				img_tile = np.expand_dims(img_tile, axis=0)
				img_tile = preprocess_func(img_tile)

				# Not sure what is happening here?
				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
				y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

				tile_counter += 1

				yield np.copy(img_tile), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_tile, img_debug, best_anchor_for_bbox, n_pos

			if C.include_full_img:

				if train_mode and C.balanced_classes and sample_selector.skip_tile_for_balanced_class(img_data):
					continue

				# IF TRAINING MODE NO RANDOM TYPE?
				img = get_image(img_data['filepath'], C.img_types, random_type=C.use_img_type)
				img_data_full = copy.deepcopy(img_data)

				if train_mode:
					img_data_full, img_full = augmentation.augment(img_data_full, img, C, augment=True, verbose=verbose)
				else:
					img_data_full, img_full = augmentation.augment(img_data_full, img, C, augment=False, verbose=verbose)

				width = img_data_full['width']
				height = img_data_full['height']
				rows = img_full.shape[0]
				cols = img_full.shape[1]

				assert cols == width
				assert rows == height

				# Get new image dimensions.
				width_resized, height_resized = get_new_img_size(width, height, C.img_size)

				# Resize image.
				img_full = cv2.resize(
					img_full,
					(width_resized, height_resized), 
					interpolation=cv2.INTER_CUBIC
				)

				img_debug = img_full.copy()

				try:
					y_rpn_cls, y_rpn_regr, best_anchor_for_bbox, n_pos = calc_region_props(
						C,
						img_data_full,
						width,
						height,
						width_resized,
						height_resized,
						get_feat_map_size,
						verbose=verbose
					)
				except Exception as e:
					print('')
					traceback.print_exc()
					print('')
					continue

				# Zero-center by mean pixel and preprocess image.
				img_full = img_full[:, :, (2, 1, 0)] # BGR -> RGB
				img_full = img_full.astype(np.float32)

				img_full = np.expand_dims(img_full, axis=0)
				img_full = preprocess_func(img_full)

				# Not sure what is happening here?
				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
				y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

				yield np.copy(img_full), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_full, img_debug, best_anchor_for_bbox, n_pos


		if train_mode == False:
			raise StopIteration

def calc_region_props(C, img_data, width, height, width_resized, height_resized, get_feat_map_size, verbose=False):

	'''

		Calculate the rpn for all anchors .
		If feature map has shape 38x50=1900, there are 1900x9=17100 potential anchors

		Args:
			C: config.
			img_data: augmented image data.
			width: original image width.
			height: original image height.
			width_resized: resized image width according to C.im_size.
			height_resized: resized image height according to C.im_size.
			get_feat_map_size: Function to calculate final layers feature map (of base model) 
				size according to input image size.
		Returns:
			y_rpn_cls: list(num_boxxes, y_is_box_valid + y_rpn_overlap).
				y_is_box_valid: 0 or 1 (0 means box is invalid, 1 means the box is valid).
				y_rpn_overlap: 0 or 1 (0 means the box is not an object, 1 means the box is an object).
			y_rpn_regr: list(num_bboxes, 4*y_rpn_overlap + y_rpn_regr).
				y_rpn_regr: x1,y1,x2,y2 bounding boxes coordinates.
			n_pos: Number of positive region proposals.

	'''

	if verbose:
		time_start = time.time()
		print('Calculating Region Proposals.')


	downscale = float(C.rpn_stride)
	anchor_sizes = C.anchor_box_scales
	anchor_ratios = C.anchor_box_ratios
	n_anch_ratios = len(anchor_ratios)
	n_anchors = len(anchor_sizes) * n_anch_ratios

	# Calculate the output map size based on network architecture.
	feature_map_width, feature_map_height = get_feat_map_size(width_resized, height_resized)

	# Initialize empty output objectives.
	y_rpn_overlap = np.zeros((feature_map_height, feature_map_width, n_anchors))
	y_is_box_valid = np.zeros((feature_map_height, feature_map_width, n_anchors))
	y_rpn_regr = np.zeros((feature_map_height, feature_map_width, 4 * n_anchors))

	n_bboxes = len(img_data['bboxes'])

	n_anchors_for_bbox = np.zeros(n_bboxes).astype(int)
	best_anchor_for_bbox = -1 * np.ones((n_bboxes, 4)).astype(int)
	best_iou_for_bbox = np.zeros(n_bboxes).astype(np.float32)
	best_x_for_bbox = np.zeros((n_bboxes, 4)).astype(int)
	best_dx_for_bbox = np.zeros((n_bboxes, 4)).astype(np.float32)

	# Get the GT box coordinates and resize to account for image resizing.
	gt_box_coords = np.zeros((n_bboxes, 4))
	for bbox_idx, bbox in enumerate(img_data['bboxes']):
		gt_box_coords[bbox_idx, 0] = bbox['x1'] * (width_resized / float(width))
		gt_box_coords[bbox_idx, 1] = bbox['x2'] * (width_resized / float(width))
		gt_box_coords[bbox_idx, 2] = bbox['y1'] * (height_resized / float(height))
		gt_box_coords[bbox_idx, 3] = bbox['y2'] * (height_resized / float(height))

	# GT for Region Proposals.
	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anch_ratios):

			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

			for ix in range(feature_map_width):

				# x-coord of the current anchor box.
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2

				# Ignore boxes that go across image boundaries.
				if x1_anc < 0 or x2_anc > width_resized:
					continue

				for jy in range(feature_map_height):

					# y-coord of the current anchor box.
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					if y1_anc < 0 or y2_anc > height_resized:
						continue

					# Indicates whether an anchor should be a target.
					bbox_type = 'neg'

					# This is the best IoU for the (x,y) coord and the current anchor.
					# Note that this is different from the best IoU for a GT bbox.
					best_iou_for_loc = 0.0

					for bbox_idx in range(n_bboxes):

						# Get IoU of the current GT box and the current anchor box.
						curr_iou = iou(
							[
								gt_box_coords[bbox_idx, 0],
								gt_box_coords[bbox_idx, 2],
								gt_box_coords[bbox_idx, 1],
								gt_box_coords[bbox_idx, 3]
							],
							[
								x1_anc,
								y1_anc,
								x2_anc,
								y2_anc
							]
						)

						# Calculate the regression targets if they will be needed.
						if curr_iou > best_iou_for_bbox[bbox_idx] or curr_iou > C.rpn_max_overlap:

							cx = (gt_box_coords[bbox_idx, 0] + gt_box_coords[bbox_idx, 1]) / 2.0
							cy = (gt_box_coords[bbox_idx, 2] + gt_box_coords[bbox_idx, 3]) / 2.0

							cx_anc = (x1_anc + x2_anc) / 2.0
							cy_anc = (y1_anc + y2_anc) / 2.0

							# x,y are the center points of GT Bbox.
							# xa,xy are the center points of anchor-box.
							# w,h are the width and height of GT Bbox.
							# wa,ha are the width and height of anchor-box.
							# tx = (x - xa) / wa
							# ty = (y - ya) / ha
							# tw = log(w / wa)
							# th = log(h / ha)

							tx = (cx - cx_anc) / (x2_anc - x1_anc)
							ty = (cy - cy_anc) / (y2_anc - y1_anc)
							tw = np.log((gt_box_coords[bbox_idx, 1] - gt_box_coords[bbox_idx, 0]) / (x2_anc - x1_anc))
							th = np.log((gt_box_coords[bbox_idx, 3] - gt_box_coords[bbox_idx, 2]) / (y2_anc - y1_anc))

						# If class is not background.
						if img_data['bboxes'][bbox_idx]['class'] != 'bg':

							# All GT boxes should be mapped to an anchor box,
							# so we keep track of which anchor box was best.

							if curr_iou > best_iou_for_bbox[bbox_idx]:

								best_anchor_for_bbox[bbox_idx] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
								best_iou_for_bbox[bbox_idx] = curr_iou
								best_x_for_bbox[bbox_idx, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
								best_dx_for_bbox[bbox_idx, :] = [tx, ty, tw, th]

							# We set the anchor to positive if the IoU is > 0.7.
							# It does not matter if there was another better box, it just indicates overlap.
							if curr_iou > C.rpn_max_overlap:

								bbox_type = 'pos'
								n_anchors_for_bbox[bbox_idx] += 1

								# Update the regression layer target if this IoU is the best for the current (x,y) and anchor position.
								if curr_iou > best_iou_for_loc:

									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th)

							# If the IoU is > 0.3 and < 0.7, it is ambiguous and no included in the objective.
							# Gray zone between neg and pos.
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:

								if bbox_type != 'pos':
									bbox_typ = 'neutral'

						# Turn on or off outputs depending on IoUs.
						if bbox_type == 'neg':

							y_is_box_valid[jy, ix, anchor_ratio_idx + n_anch_ratios * anchor_size_idx] = 1
							y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anch_ratios * anchor_size_idx] = 0

						elif bbox_type == 'neutral':

							y_is_box_valid[jy, ix, anchor_ratio_idx + n_anch_ratios * anchor_size_idx] = 0
							y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anch_ratios * anchor_size_idx] = 0

						elif bbox_type == 'pos':

							y_is_box_valid[jy, ix, anchor_ratio_idx + n_anch_ratios * anchor_size_idx] = 1
							y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anch_ratios * anchor_size_idx] = 1
							start = 4 * (anchor_ratio_idx + n_anch_ratios * anchor_size_idx)
							y_rpn_regr[jy, ix, start:start+4] = best_regr

	# Ensure that every bbox has at least one positive RPN region.
	for idx in range(n_anchors_for_bbox.shape[0]):
		if n_anchors_for_bbox[idx] == 0:
			
			# No box with an IOU greater than zero.
			if best_anchor_for_bbox[idx, 0] == -1:
				continue

			y_is_box_valid[
				best_anchor_for_bbox[idx, 0],
				best_anchor_for_bbox[idx, 1],
				best_anchor_for_bbox[idx, 2] + n_anch_ratios * best_anchor_for_bbox[idx, 3]
			] = 1

			y_rpn_overlap[
				best_anchor_for_bbox[idx, 0],
				best_anchor_for_bbox[idx, 1],
				best_anchor_for_bbox[idx, 2] + n_anch_ratios * best_anchor_for_bbox[idx, 3]
			] = 1

			start = 4 * (best_anchor_for_bbox[idx, 2] + n_anch_ratios * best_anchor_for_bbox[idx, 3])

			y_rpn_regr[
				best_anchor_for_bbox[idx, 0],
				best_anchor_for_bbox[idx, 1],
				start:start+4
			] = best_dx_for_bbox[idx, :]

	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

	n_pos = len(pos_locs[0])
	n_neg = len(neg_locs[0])

	# One issue is that the RPN has many more negative than positive regions.
	# Turn off some of the negative regions and limit it to 256 regions.
	max_n_regions = 256

	if n_pos > max_n_regions/2:

		pos_unique, pos_counts = np.unique(neg_locs[0], return_counts=True)
		pos_probs = pos_counts / n_pos

		pos_counts = dict(zip(pos_unique, pos_counts))
		pos_probs = dict(zip(pos_unique, pos_probs))

		probs = [pos_probs[l]/pos_counts[l] for l in pos_locs[0]]

		val_locs = np.random.choice(n_pos, n_pos - int(max_n_regions/2), replace=False, p=probs)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0

		n_pos = int(max_n_regions/2)

	if n_neg + n_pos > max_n_regions:

		neg_unique, neg_counts = np.unique(neg_locs[0], return_counts=True)
		neg_probs = neg_counts / n_neg

		neg_counts = dict(zip(neg_unique, neg_counts))
		neg_probs = dict(zip(neg_unique, neg_probs))

		probs = [neg_probs[l]/neg_counts[l] for l in neg_locs[0]]

		val_locs = np.random.choice(n_neg, n_neg - n_pos, replace=False, p=probs)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

	if verbose:
		time_end = time.time()
		print('Execution Time for Calculating Region Proposals: ' + ms_output(time_end - time_start))

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr), best_anchor_for_bbox, n_pos