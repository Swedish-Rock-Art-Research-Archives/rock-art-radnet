
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed
from keras.applications.resnet50 import preprocess_input

from ..RoiPoolingConv import RoiPoolingConv
from ..FixedBatchNormalization import FixedBatchNormalization

"""
	ResNet50 model for Hard.
	[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
"""

FINE_TUNING_CUT = 38

WEIGHT_PATH = 'faster_rcnn/base_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def get_img_output_length(width, height):
	
	def get_output_length(input_length):

		# zero_pad
		input_length += 6

		# apply 4 strided convolutions
		filter_sizes = [7, 3, 1, 1]
		stride = 2

		for filter_size in filter_sizes:
			input_length = (input_length - filter_size + stride) // stride

		return input_length

	return get_output_length(width), get_output_length(height)

def preprocess(img):

	return preprocess_input(img)

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):

	nb_filter1, nb_filter2, nb_filter3 = filters
	bn_axis = 3

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)

	return x

def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):

	# identity block time distributed

	nb_filter1, nb_filter2, nb_filter3 = filters
	bn_axis = 3

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal', padding='same'), name=conv_name_base + '2b')(x)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)

	return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):

	nb_filter1, nb_filter2, nb_filter3 = filters
	bn_axis = 3


	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
	x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
	shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = Add()([x, shortcut])
	x = Activation('relu')(x)

	return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

	# conv block time distributed

	nb_filter1, nb_filter2, nb_filter3 = filters
	bn_axis = 3

	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
	x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

	shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
	shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

	x = Add()([x, shortcut])
	x = Activation('relu')(x)
	
	return x


def nn_base(input_tensor=None, trainable=False, weights='imagenet'):

	# Define input tensor.
	if input_tensor == None:
		input_tensor = Input(shape=(None, None, 3))

	bn_axis = 3

	'''
	# Read base model with pretrained weights.
	base_model = ResNet50(
		include_top=False,
		weights=weights,
		input_tensor=input_tensor,
	)

	# Define new output layer.
	model = Model(
		inputs=base_model.input,
		outputs=base_model.layers[141].output
	)

	# Make all layers trainable.
	for layer in model.layers:
		layer.trainable = trainable

	for i, layer in enumerate(model.layers):
		print(i, layer.name, layer.output_shape)

	import sys
	sys.exit(1)
	'''

	x = ZeroPadding2D((3, 3))(input_tensor)

	x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
	x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable) # +12
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=trainable) # + 10
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)

	x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
	x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)

	model = Model(
		inputs=input_tensor,
		outputs=x
	)

	if weights != None:
		
		model.load_weights(WEIGHT_PATH, by_name=True)
		'''
		for layer in model.layers:
			layer.trainable = trainable
		'''
		for layer in model.layers[:FINE_TUNING_CUT]:
			layer.trainable = False

		for layer in model.layers[FINE_TUNING_CUT:]:
			layer.trainable = trainable
		
	'''
	for i, layer in enumerate(model.layers):
		print(i, layer.name, layer.output_shape)
	'''
	return model.layers[-1].output


def classifier_layer(input_layer, input_rois, n_rois, nb_classes=4):

	"""

	Create a classifier layer
	
	Args:
		input_layer: resnet50
		input_rois: `(1,n_rois,4)` list of rois, with ordering (x,y,w,h)
		n_rois: number of rois to be processed in one time (4 in here)

	Returns:
		list(out_class, out_regr)
		out_class: classifier layer output
		out_regr: regression layer output

	"""

	pooling_regions = 14
	input_shape = (n_rois, 14, 14, 1024)

	out_roi_pool = RoiPoolingConv(
		pooling_regions,
		n_rois
	)([input_layer, input_rois])

	out = conv_block_td(out_roi_pool, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=True)
	out = identity_block_td(out, 3, [512, 512, 2048], stage=5, block='b', trainable=True)
	out = identity_block_td(out, 3, [512, 512, 2048], stage=5, block='c', trainable=True)
	out = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(out)
	out = TimeDistributed(Flatten())(out)

	out_class = TimeDistributed(
		Dense(
			nb_classes,
			activation='softmax',
			kernel_initializer='zero'
		),
		name='dense_class_{}'.format(nb_classes)
	)(out)

	out_regr = TimeDistributed(
		Dense(
			4 * (nb_classes-1),
			activation='linear',
			kernel_initializer='zero'
		),
		name='dense_regress_{}'.format(nb_classes)
	)(out)

	return [out_class, out_regr]