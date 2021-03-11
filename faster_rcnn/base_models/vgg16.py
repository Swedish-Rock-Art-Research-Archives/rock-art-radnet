
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import TimeDistributed
from keras.applications.vgg16 import preprocess_input

from ..RoiPoolingConv import RoiPoolingConv

"""
	VGG16 model for Keras.
	[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""

FINE_TUNING_CUT = 7

def get_img_output_length(width, height):
	
	def get_output_length(input_length):
		return input_length//16

	return get_output_length(width), get_output_length(height)

def preprocess(img):

	return preprocess_input(img)

def nn_base(input_tensor=None, trainable=False, weights='imagenet'):

	# Define input tensor.
	if input_tensor == None:
		input_tensor = Input(shape=(None, None, 3))

	# Read base model with pretrained weights.
	base_model = VGG16(
		include_top=False,
		weights=weights,
		input_tensor=input_tensor,
	)

	# Define new output layer.
	model = Model(
		inputs=base_model.input,
		outputs=base_model.get_layer('block5_conv3').output
	)

	# Make all layers trainable.
	#for layer in model.layers:
	#	layer.trainable = trainable

	
	# May some layers trainable.

	for layer in model.layers[:FINE_TUNING_CUT]:
		layer.trainable = False

	for layer in model.layers[FINE_TUNING_CUT:]:
		layer.trainable = trainable
	

	#for i, layer in enumerate(model.layers):
	#	print(i, layer.name, layer.output_shape, layer.trainable)

	return model.layers[-1].output

def classifier_layer(input_layer, input_rois, n_rois, nb_classes=4):

	"""

	Create a classifier layer
	
	Args:
		input_layer: vgg
		input_rois: `(1,n_rois,4)` list of rois, with ordering (x,y,w,h)
		n_rois: number of rois to be processed in one time (4 in here)

	Returns:
		list(out_class, out_regr)
		out_class: classifier layer output
		out_regr: regression layer output

	"""

	pooling_regions = 7
	#input_shape = (n_rois, 7, 7, 512)

	# out_roi_pool.shape = (1, n_rois, channels, pool_size, pool_size)
	# n_rois (4) 7x7 roi pooling
	out_roi_pool = RoiPoolingConv(
		pooling_regions,
		n_rois
	)([input_layer, input_rois])

	# Flatten the convolutional layer and connected to 2 FC and 2 dropout
	out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
	out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
	out = TimeDistributed(Dropout(0.5))(out)
	out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
	out = TimeDistributed(Dropout(0.5))(out)

	# There are two output layer
	# out_class: softmax acivation function for classify the class name of the object
	# out_regr: linear activation function for bboxes coordinates regression
	out_class = TimeDistributed(
		Dense(
			nb_classes,
			activation='softmax',
			kernel_initializer='zero'
		),
		name='dense_class_{}'.format(nb_classes)
	)(out)
	
	# note: no regression target for bg class
	out_regr = TimeDistributed(
		Dense(
			4 * (nb_classes-1),
			activation='linear',
			kernel_initializer='zero'
		),
		name='dense_regress_{}'.format(nb_classes)
	)(out)

	return [out_class, out_regr]