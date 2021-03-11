from __future__ import generator_stop


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
import random
import urllib.request

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, SGD, Nadam
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


SEED = 64

MODELS_PATH = 'models'
MODEL_NAME = 'raod_base' # Leave to None if you want to automatically generate name.

TRAIN_ANNOT_PATH = 'data/train.csv'
TRAIN_DATA_PATH = 'data/train'

VAL_ANNOT_PATH = 'data/val.csv'
VAL_DATA_PATH = 'data/val'

EPOCH_LENGTH = 173
N_EPOCHS = 100
USE_VALIDATION = True

def ms_output(seconds):
	
	return str(pd.to_timedelta(seconds, unit='s'))

def silly_name_gen():

	word_url = "http://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain"
	response = urllib.request.urlopen(word_url)
	long_txt = response.read().decode()
	words = long_txt.splitlines()

	upper_words = [word for word in words if word[0].isupper()]
	name_words  = [word for word in upper_words if not word.isupper()]

	return '_'.join([name_words[random.randint(0, len(name_words))] for i in range(2)])

def write_log(callback, names, logs, batch_no):
	for name, value in zip(names, logs):
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		callback.writer.add_summary(summary, batch_no)
		callback.writer.flush()

def create_model_folder(model_name):

	model_path = os.path.join(MODELS_PATH, model_name)

	if os.path.exists(model_path):
		shutil.rmtree(model_path)

	os.makedirs(model_path)
	os.makedirs(os.path.join(model_path, 'viz'))
	os.makedirs(os.path.join(model_path, 'test'))
	#os.makedirs(os.path.join(model_path, 'test_rpn'))

def get_selected_samples(Y1, C):

	# Find out the positive anchors and negative anchors
	neg_samples = np.where(Y1[0, :, -1] == 1)
	pos_samples = np.where(Y1[0, :, -1] == 0)

	if len(neg_samples) > 0:
		neg_samples = neg_samples[0]
	else:
		neg_samples = []

	if len(pos_samples) > 0:
		pos_samples = pos_samples[0]
	else:
		pos_samples = []

	# If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
	if len(pos_samples) < C.n_rois//2:
		selected_pos_samples = pos_samples.tolist()
	else:
		selected_pos_samples = np.random.choice(pos_samples, C.n_rois//2, replace=False).tolist()

	# Randomly choose (n_rois - num_pos) neg samples
	if len(neg_samples) > 0:
		try:
			selected_neg_samples = np.random.choice(neg_samples, C.n_rois - len(selected_pos_samples), replace=False).tolist()
		except:
			selected_neg_samples = np.random.choice(neg_samples, C.n_rois - len(selected_pos_samples), replace=True).tolist()

		return selected_pos_samples + selected_neg_samples, len(pos_samples)

	else:

		selected_pos_samples = np.random.choice(pos_samples, len(pos_samples), replace=False).tolist()
		selected_pos_samples += np.random.choice(pos_samples, C.n_rois - len(selected_pos_samples), replace=True).tolist()

		return selected_pos_samples, len(pos_samples)

def main():

	# Set seed.
	np.random.seed(SEED)
	tf.set_random_seed(SEED)

	# Read config.
	C = config.Config()
	class_mapping = C.class_mapping

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


	# Read validation data.
	if USE_VALIDATION:
		data_val, _, _ = get_data(VAL_ANNOT_PATH, VAL_DATA_PATH, C.img_types)

	# Create paths.
	if MODEL_NAME != None:
		#model_name = C.model_path + '_' + datetime.today().strftime('%y%m%d') + '_' + MODEL_NAME
		model_name = C.model_path + '_' + MODEL_NAME
		if os.path.exists(os.path.join(MODELS_PATH, model_name)):
			print('Model already exist.')
			sys.exit(1)
	else:
		model_name = C.model_path + '_' + datetime.today().strftime('%y%m%d') + '_' + silly_name_gen()
		
	model_path = os.path.join(MODELS_PATH, model_name)
	weights_path = os.path.join(model_path, 'weights.hdf5')
	config_path = os.path.join(model_path, 'config.pickle')
	config_json_path = os.path.join(model_path, 'config.json')
	record_path = os.path.join(model_path, 'record.csv')

	C.weights_path = weights_path
	
	# Create model folder.
	create_model_folder(model_name)

	# Save config.
	with open(config_path, 'wb') as f:
		pickle.dump(C, f)

	with open(config_json_path, 'w') as f:
		json.dump(C.__dict__, f, indent=4)
	
	# Create generators.
	data_train_gen = get_tile_generator(data_train, C, base_model.get_img_output_length, class_count, base_model.preprocess, train_mode=True, verbose=False)
	
	if USE_VALIDATION:
		data_val_gen = get_tile_generator(data_val, C, base_model.get_img_output_length, class_count, base_model.preprocess, train_mode=False, verbose=False)

	# Define shapes.
	img_input_shape = (None, None, 3)
	img_input = Input(shape=img_input_shape)
	roi_input = Input(shape=(None, 4))
	n_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

	# Define base network with shared layers, RPN and classifier.
	base_net_output_layer = base_model.nn_base(img_input, trainable=C.base_net_trainable, weights=C.base_net_weights)
	rpn = rpn_layer(base_net_output_layer, n_anchors)
	classifier = base_model.classifier_layer(
		base_net_output_layer,
		roi_input,
		C.n_rois,
		nb_classes=len(class_mapping)
	)

	# Create models.
	model_rpn = Model(img_input, rpn[:2])
	model_classifier = Model([img_input, roi_input], classifier)
	model_all = Model([img_input, roi_input], rpn[:2] + classifier)

	# Create var for recording training process.
	df_record = pd.DataFrame(
		columns=[
			'elapsed_time',
			'mean_overlapping_bboxes',
			'val_mean_overlapping_bboxes',
			'loss_rpn_cls',
			'val_loss_rpn_cls',
			'loss_rpn_regr',
			'val_loss_rpn_regr',
			'loss_detector_cls',
			'val_loss_detector_cls',
			'loss_detector_regr',
			'val_loss_detector_regr',
			'total_loss',
			'val_total_loss',
			'detector_acc',
			'val_detector_acc',
			'model_improvement'
		]
	)

	# Compile models.
	model_rpn.compile(
		optimizer=Adam(lr=1e-5 * 5.0), #SGD(lr=1e-3, momentum=0.9, decay=0.0005),
		loss=[
			rpn_loss_cls(n_anchors),
			rpn_loss_regr(n_anchors)
		]
	)
	model_classifier.compile(
		optimizer=Adam(lr=1e-5 * 5.0), #SGD(lr=1e-3, momentum=0.9, decay=0.0005),
		loss=[
			class_loss_cls,
			class_loss_regr(len(class_mapping)-1)
		],
		metrics={
			'dense_class_{}'.format(len(class_mapping)): 'accuracy'
		}
	)
	model_all.compile(
		optimizer='sgd',
		loss='mae'
	)

	# Setup Tensorboard.
	callback = TensorBoard(model_path)
	callback.set_model(model_all)

	# Training settings.
	iter_num = 0
	train_step = 0
	losses = np.zeros((EPOCH_LENGTH, 5))
	rpn_accuracy_for_epoch = []
	best_total_loss = np.Inf

	# Start training.
	start_time = time.time()
	print('\n\nStart training.')

	for epoch_num in range(N_EPOCHS):

		pbar = generic_utils.Progbar(EPOCH_LENGTH)
		print('Epoch {}/{}'.format(epoch_num + 1, N_EPOCHS))

		while True:

			# Get next batch (image).
			img, Y, img_data, img_debug, best_anchor_for_bbox, debug_n_pos = next(data_train_gen)

			# If no GT boxes.
			if len(img_data['bboxes']) == 0:
				continue

			# Train on batch.
			loss_rpn = model_rpn.train_on_batch(img, Y)

			# Get predicted RPN from RPN model [rpn_cls, rpn_regr].
			P_rpn = model_rpn.predict_on_batch(img)

			'''
			if iter_num == 0:

				colormap = [
					((166,206,227), (31,120,180)), # Light Blue, Blue
					((178,223,138), (51,160,44)), # Light Green, Green
					((251,154,153), (227,26,28)), # Light Red, Red
					((253,191,111), (255,127,0)), # Light Orange, Orange
					((202,178,214), (106,61,154)), # Light Purple, Purple
				]

				img_debug = cv2.cvtColor(img_debug, cv2.COLOR_BGR2GRAY)
				img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2RGB)

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

					cv2.circle(img_debug, center, 3, color, -1)
					cv2.rectangle(
						img_debug,
						(center[0]-int(anchor_width/2), center[1]-int(anchor_height/2)),
						(center[0]+int(anchor_width/2), center[1]+int(anchor_height/2)),
						color,
						2
					)

				plt.figure(figsize=(8,8))
				plt.imshow(img_debug)
				plt.title(img_data['filepath'])
				plt.show()

				for i in range(9):

					fig, ax = plt.subplots()
					im = ax.imshow(Y[0][0, :, :, i])
					plt.colorbar(im)
					plt.title('isValid' + str(i))
					plt.show()

					fig, ax = plt.subplots()
					im = ax.imshow(Y[0][0, :, :, i+9])
					plt.colorbar(im)
					plt.title('isObject' + str(i))
					plt.show()

					fig, ax = plt.subplots()
					im = ax.imshow(P_rpn[0][0, :, :, i])
					plt.colorbar(im)
					plt.title('Prediction' + str(i))
					plt.show()
			'''


			# R: bboxes (shape=(300,4))
			# Convert RPN layer to ROI bboxes.
			R = rpn_to_roi(
				P_rpn[0],
				P_rpn[1],
				C,
				use_regr=True,
				overlap_thresh=0.7,
				max_boxes=300
			)

			# Note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
			# X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes.
			# Y1: one hot code for bboxes from above => x_roi (X)
			# Y2: corresponding labels and corresponding gt bboxes
			X2, Y1, Y2, IouS = calc_iou(R, img_data, C, class_mapping)

			# If X2 is None means there are no matching bboxes
			if X2 is None:
				rpn_accuracy_for_epoch.append(0)
				continue


			sel_samples, n_pos_samples = get_selected_samples(Y1, C)
			rpn_accuracy_for_epoch.append(n_pos_samples)

			# training_data: [img, X2[:, sel_samples, :]]
			# labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
			#  img                     => img_data resized image
			#  X2[:, sel_samples, :] => n_rois (4 in here) bboxes which contains selected neg and pos
			#  Y1[:, sel_samples, :] => one hot encode for n_rois bboxes which contains selected neg and pos
			#  Y2[:, sel_samples, :] => labels and gt bboxes for n_rois bboxes which contains selected neg and pos

			loss_detector = model_classifier.train_on_batch(
				[
					img,
					X2[:, sel_samples, :]
				],
				[
					Y1[:, sel_samples, :],
					Y2[:, sel_samples, :]
				]
			)

			# Log losses.
			losses[iter_num, 0] = loss_rpn[1]
			losses[iter_num, 1] = loss_rpn[2]

			write_log(
				callback,
				['rpn_cls_loss', 'rpn_reg_loss'],
				[loss_rpn[1], loss_rpn[2]],
				train_step
			)

			losses[iter_num, 2] = loss_detector[1]
			losses[iter_num, 3] = loss_detector[2]
			losses[iter_num, 4] = loss_detector[3]
			
			write_log(
				callback,
				['detector_cls_loss', 'detector_reg_loss', 'detector_acc'],
				[loss_detector[1], loss_detector[2], loss_detector[3]],
				train_step
			)

			iter_num += 1
			train_step += 1

			pbar.update(
				iter_num,
				[
					('rpn_cls', losses[iter_num-1, 0]),
					('rpn_regr', losses[iter_num-1, 1]),
					('detector_cls', losses[iter_num-1, 2]),
					('detector_regr', losses[iter_num-1, 3])
				]
			)

			if iter_num == EPOCH_LENGTH:

				# Compute epoch losses.
				loss_rpn_cls = np.mean(losses[:, 0])
				loss_rpn_regr = np.mean(losses[:, 1])
				loss_detector_cls = np.mean(losses[:, 2])
				loss_detector_regr = np.mean(losses[:, 3])
				class_acc = np.mean(losses[:, 4])

				mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
				rpn_accuracy_for_epoch = []

				elapsed_time = (time.time() - start_time)
				curr_total_loss = loss_rpn_cls + loss_rpn_regr + loss_detector_cls + loss_detector_regr
				iter_num = 0

				if C.verbose:

					print('')

					if mean_overlapping_bboxes == 0:
						print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
					else:
						print('(TRAINING) Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					
					print('(TRAINING) Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('(TRAINING) Loss RPN regression: {}'.format(loss_rpn_regr))
					print('(TRAINING) Loss Detector classifier: {}'.format(loss_detector_cls))
					print('(TRAINING) Loss Detector regression: {}'.format(loss_detector_regr))
					print('(TRAINING) Detector accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('(TRAINING) Total Loss: {}'.format(curr_total_loss))
					print('Elapsed time: ' + ms_output(elapsed_time))
					print('')


				# Validation.
				record_row = {}
				if USE_VALIDATION:

					val_start_time = time.time()
					print('\nPerforming Validation.')

					val_rpn_accuracy = []
					val_rpn_cls_loss = []
					val_rpn_reg_loss = []
					val_detector_cls_loss = []
					val_detector_reg_loss = []
					val_detector_acc = []

					while True:

						try:
							img_val, Y_val, img_data_val, _, _, _ = next(data_val_gen)

							# Validate on batch.
							val_loss_rpn = model_rpn.test_on_batch(img_val, Y_val)

							P_rpn_val = model_rpn.predict_on_batch(img_val)
							R_val = rpn_to_roi(
								P_rpn_val[0],
								P_rpn_val[1],
								C,
								use_regr=True,
								overlap_thresh=0.7,
								max_boxes=300
							)

							X2_val, Y1_val, Y2_val, _ = calc_iou(R_val, img_data_val, C, class_mapping)

							if X2_val is None:
								continue

							val_sel_samples, val_n_pos_samples = get_selected_samples(Y1_val, C)
							
							val_loss_detector = model_classifier.test_on_batch(
								[
									img_val,
									X2_val[:, val_sel_samples, :]
								],
								[
									Y1_val[:, val_sel_samples, :],
									Y2_val[:, val_sel_samples, :]
								]
							)

							val_rpn_accuracy.append(val_n_pos_samples)
							val_rpn_cls_loss.append(val_loss_rpn[1])
							val_rpn_reg_loss.append(val_loss_rpn[2])
							val_detector_cls_loss.append(val_loss_detector[1])
							val_detector_reg_loss.append(val_loss_detector[2])
							val_detector_acc.append(val_loss_detector[3])
							
						except RuntimeError:
							break	
						except StopIteration:
							break	

						except:
							print(traceback.print_exc())
							sys.exit(1)

					data_val_gen = get_tile_generator(data_val, C, base_model.get_img_output_length, class_count, base_model.preprocess, train_mode=False, verbose=False)
					val_mean_overlapping_bboxes = float(sum(val_rpn_accuracy)) / len(val_rpn_accuracy)
					val_total_loss = np.mean(val_rpn_cls_loss) + np.mean(val_rpn_reg_loss) + np.mean(val_detector_cls_loss) + np.mean(val_detector_reg_loss)

					print('(VALIDATION) Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(val_mean_overlapping_bboxes))
					
					print('(VALIDATION) Mean Loss RPN classifier: {}'.format(np.mean(val_rpn_cls_loss)))
					print('(VALIDATION) Mean Loss RPN regression: {}'.format(np.mean(val_rpn_reg_loss)))
					print('(VALIDATION) Mean Loss Detector classifier: {}'.format(np.mean(val_detector_cls_loss)))
					print('(VALIDATION) Mean Loss Detector regression: {}'.format(np.mean(val_detector_reg_loss)))
					print('(VALIDATION) Mean Detector accuracy for bounding boxes from RPN: {}'.format(np.mean(val_detector_acc)))
					print('(VALIDATION) Total Loss: {}'.format(val_total_loss))

					record_row['val_mean_overlapping_bboxes'] = round(val_mean_overlapping_bboxes, 3)
					record_row['val_detector_acc'] = round(np.mean(val_detector_acc), 3)
					record_row['val_loss_rpn_cls'] = round(np.mean(val_rpn_cls_loss), 3)
					record_row['val_loss_rpn_regr'] = round(np.mean(val_rpn_reg_loss), 3)
					record_row['val_loss_detector_cls'] = round(np.mean(val_detector_cls_loss), 3)
					record_row['val_loss_detector_regr'] = round(np.mean(val_detector_reg_loss), 3)
					record_row['val_total_loss'] = round(val_total_loss, 3)

					val_elapsed_time = (time.time() - val_start_time)
					print('Validation execution time: ' + ms_output(val_elapsed_time))
					print('')

					if val_total_loss < best_total_loss:

						record_row['model_improvement'] = val_total_loss - best_total_loss
						
						if C.verbose:
							print('Total loss decreased from {} to {}, saving weights'.format(best_total_loss, val_total_loss))
							print('')

						best_total_loss = val_total_loss
						model_all.save_weights(weights_path)

					else:

						record_row['model_improvement'] = None

				else:

					record_row['val_mean_overlapping_bboxes'] = None
					record_row['val_detector_acc'] = None
					record_row['val_loss_rpn_cls'] = None
					record_row['val_loss_rpn_regr'] = None
					record_row['val_loss_detector_cls'] = None
					record_row['val_loss_detector_regr'] = None
					record_row['val_total_loss'] = None

					if curr_total_loss < best_total_loss:

						record_row['model_improvement'] = curr_total_loss - best_total_loss
						
						if C.verbose:
							print('Total loss decreased from {} to {}, saving weights'.format(best_total_loss, curr_total_loss))
							print('')

						best_total_loss = curr_total_loss
						model_all.save_weights(weights_path)

					else:

						record_row['model_improvement'] = None


				# Log epoch averages.
				write_log(
					callback,
					[
						'Elapsed_time',
						'mean_overlapping_bboxes',
						'mean_rpn_cls_loss',
						'mean_rpn_reg_loss',
						'mean_detector_cls_loss',
						'mean_detector_reg_loss',
						'mean_detector_acc',
						'total_loss'
					],
					[
						elapsed_time/60,
						mean_overlapping_bboxes,
						loss_rpn_cls,
						loss_rpn_regr,
						loss_detector_cls,
						loss_detector_regr,
						class_acc,
						curr_total_loss
					],
					epoch_num
				)

				record_row['mean_overlapping_bboxes'] = round(mean_overlapping_bboxes, 3)
				record_row['detector_acc'] = round(class_acc, 3)
				record_row['loss_rpn_cls'] = round(loss_rpn_cls, 3)
				record_row['loss_rpn_regr'] = round(loss_rpn_regr, 3)
				record_row['loss_detector_cls'] = round(loss_detector_cls, 3)
				record_row['loss_detector_regr'] = round(loss_detector_regr, 3)
				record_row['total_loss'] = round(curr_total_loss, 3)
				record_row['elapsed_time'] = round(elapsed_time/60, 3)

				df_record = df_record.append(record_row, ignore_index=True)
				df_record.to_csv(record_path, index=0)

				break
	
	print('Training Complete! Exiting.')

	fig = plt.figure(figsize=(15,5))
	plt.subplot(1,2,1)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['mean_overlapping_bboxes'], 'r', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_mean_overlapping_bboxes'], 'b', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['mean_overlapping_bboxes'].rolling(window=20).mean(), 'r', label='Train')
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_mean_overlapping_bboxes'].rolling(window=20).mean(), 'b', label='Val')
	plt.title('mean_overlapping_bboxes')
	plt.legend()
	plt.subplot(1,2,2)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['detector_acc'], 'r', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_detector_acc'], 'b', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['detector_acc'].rolling(window=20).mean(), 'r', label='Train')
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_detector_acc'].rolling(window=20).mean(), 'b', label='Val')
	plt.title('class_acc')
	plt.legend()
	fig.savefig(os.path.join(model_path, 'viz/accuracy.png'))

	fig = plt.figure(figsize=(15,5))
	plt.subplot(1,2,1)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['loss_rpn_cls'], 'r', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_loss_rpn_cls'], 'b', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['loss_rpn_cls'].rolling(window=20).mean(), 'r', label='Train')
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_loss_rpn_cls'].rolling(window=20).mean(), 'b', label='Val')
	plt.title('loss_rpn_cls')
	plt.legend()
	plt.subplot(1,2,2)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['loss_rpn_regr'], 'r', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_loss_rpn_regr'], 'b', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['loss_rpn_regr'].rolling(window=20).mean(), 'r', label='Train')
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_loss_rpn_regr'].rolling(window=20).mean(), 'b', label='Val')
	plt.title('loss_rpn_regr')
	plt.legend()
	fig.savefig(os.path.join(model_path, 'viz/rpn_loss.png'))

	fig = plt.figure(figsize=(15,5))
	plt.subplot(1,2,1)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['loss_detector_cls'], 'r', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_loss_detector_cls'], 'b', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['loss_detector_cls'].rolling(window=20).mean(), 'r', label='Train')
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_loss_detector_cls'].rolling(window=20).mean(), 'b', label='Val')
	plt.title('loss_detector_cls')
	plt.legend()
	plt.subplot(1,2,2)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['loss_detector_regr'], 'r', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_loss_detector_regr'], 'b', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['loss_detector_regr'].rolling(window=20).mean(), 'r', label='Train')
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_loss_detector_regr'].rolling(window=20).mean(), 'b', label='Val')

	plt.title('loss_detector_regr')
	plt.legend()
	fig.savefig(os.path.join(model_path, 'viz/detector_loss.png'))

	fig = plt.figure(figsize=(16,8))
	plt.plot(np.arange(0, df_record.shape[0]), df_record['total_loss'], 'r', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_total_loss'], 'b', alpha=0.3)
	plt.plot(np.arange(0, df_record.shape[0]), df_record['total_loss'].rolling(window=20).mean(), 'r', label='Train')
	plt.plot(np.arange(0, df_record.shape[0]), df_record['val_total_loss'].rolling(window=20).mean(), 'b', label='Val')
	plt.title('total_loss')
	plt.legend()
	fig.savefig(os.path.join(model_path, 'viz/total_loss.png'))

	
	
if __name__ == '__main__':

	main()

	