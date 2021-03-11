



class Config:

	def __init__(self):

		# Print the process.
		self.verbose = True

		# Name of the base network.
		# Can be any of: vgg16, resnet50
		self.network = 'resnet50'
		self.base_net_trainable = False
		self.base_net_cont_trainable = True
		self.base_net_weights = 'imagenet' # imagenet or None

		# Settings for data augmentation.
		self.use_horizontal_flips = True
		self.use_vertical_flips = True
		self.use_90_rotations = True
		self.use_rotations = True
		self.use_shear = True
		self.use_brightness = True
		self.use_noise = True

		self.use_img_type = False

		self.img_types = [
			'enhanced_topo_grey',
			'topo_grey',
		]

		# Tile settings.
		self.tile_size = 2000
		self.tile_overlap = 400
		self.tile_bbox_clip_threshold = 0.75
		self.max_n_tiles_train = 1 #10
		self.max_n_tiles_val = 1
		self.include_full_img = False


		# Anchor box scales.
		# Scale according to image size.
		# Original scales in paper: [128, 256, 512]
		self.anchor_box_scales = [64, 128, 256, 512]
		#self.anchor_box_scales = [64, 128, 256, 512]

		# Anchor box ratios.

		self.anchor_box_ratios = [
			[1.0, 1.0],
			[1.0, 2.0],
			[2.0, 1.0]
		]
		'''
		
		self.anchor_box_ratios = [
			[1.0, 1.0],
			[1.0, 2.0],
			[2.0, 1.0],
			[1.0, 3.0],
			[3.0, 1.0]
		]
		'''
		
		# Size to resize smallest side of image.
		# Original size in paper: 600
		self.img_size = 600 #1000

		# Image channel-wise mean to subtract (BGR).
		#self.img_channel_mean = [103.939, 116.779, 123.68]
		#self.img_scaling_factor = 1.0

		# Number of ROIs at once.
		self.n_rois = 20 #300

		# Stride at the RPN.
		# Depends on network configuration.
		self.rpn_stride = 16

		# Balanced classes.
		self.balanced_classes = True

		# Scaling the STD.
		self.std_scaling = 4.0
		self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

		# Overlaps for RPN.
		self.rpn_min_overlap = 0.3
		self.rpn_max_overlap = 0.7

		# Overlaps for classifier ROIs.
		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5

		# Placeholder for the class mapping.
		
		self.class_mapping = {
			'boat': 0,
			'human': 1,
			'other': 2,
			'animal': 3,
			'circle': 4,
			'wheel': 5,
			'bg': 6
		}
		
		'''
		self.class_mapping = {
			'boat': 0,
			'cupmark': 1,
			'human': 2,
			'other': 3,
			'animal': 4,
			'circle': 5,
			'footsole': 6,
			'wheel': 7,
			'wagon': 8,
			'bg': 9
		}
		'''
		'''
		self.class_mapping = {
			'orange': 0,
			'apple': 1,
			'banana': 2,
			'bg': 3
		}
		'''
		# Output model path.
		self.model_path = 'faster_rcnn_' + self.network



