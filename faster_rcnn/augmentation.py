
import cv2
import copy
import math
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import skimage as skimage

from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=1):
	return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def strap_img(img):

	finite_mask = np.isfinite(img[:,:,1])

	if np.sum(~finite_mask) == 0:
		row_idx, col_idx = np.nonzero(img[:,:,1])
	else:
		row_idx, col_idx = np.nonzero(finite_mask)

	row_min = row_idx.min()
	row_max = row_idx.max()
	col_min = col_idx.min()
	col_max = col_idx.max()

	return row_min, row_max, col_min, col_max

def clip_box(bbox, img_box, alpha):

	"""

	Clip the bounding boxes to the borders of an image

	Parameters
	----------

	bbox: numpy.ndarray
	    Numpy array containing bounding boxes of shape `N X 4` where N is the 
	    number of bounding boxes and the bounding boxes are represented in the
	    format `x1 y1 x2 y2`

	img_box: numpy.ndarray
	    An array of shape (4,) specifying the diagonal co-ordinates of the image
	    The coordinates are represented in the format `x1 y1 x2 y2`
	    
	alpha: float
	    If the fraction of a bounding box left in the image after being clipped is 
	    less than `alpha` the bounding box is dropped. 

	Returns
	-------
	['human' '7113' '5534' '7199' '5606']

	numpy.ndarray
	    Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
	    number of bounding boxes left are being clipped and the bounding boxes are represented in the
	    format `x1 y1 x2 y2` 

	"""

	mask_outside = (bbox[:, 0] > img_box[2]) | (bbox[:, 2] < img_box[0]) | (bbox[:, 1] > img_box[3]) | (bbox[:, 3] < img_box[1])

	ar_ = (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])
	x_min = np.maximum(bbox[:,0], img_box[0]).reshape(-1,1)
	y_min = np.maximum(bbox[:,1], img_box[1]).reshape(-1,1)
	x_max = np.minimum(bbox[:,2], img_box[2]).reshape(-1,1)
	y_max = np.minimum(bbox[:,3], img_box[3]).reshape(-1,1)

	bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))

	delta_area = ((ar_ - ((bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])))/ar_)

	mask_inside = (delta_area < (1 - alpha)).astype(int)
	mask = ((mask_outside == 0) & (mask_inside == 1))

	bbox = bbox[mask, :]

	return bbox, mask

def horizontal_flip(img, bboxes, verbose=False):

	if verbose:
		print('\tHorizontal Flip')

	rows, cols = img.shape[:2]

	img = cv2.flip(img, 1)
	for bbox in bboxes:
		x1 = bbox['x1']
		x2 = bbox['x2']
		bbox['x2'] = cols - x1
		bbox['x1'] = cols - x2

	return img, bboxes

def vertical_flip(img, bboxes, verbose=False):

	if verbose:
		print('\tVertical Flip')

	rows, cols = img.shape[:2]

	img = cv2.flip(img, 0)
	for bbox in bboxes:
		y1 = bbox['y1']
		y2 = bbox['y2']
		bbox['y2'] = rows - y1
		bbox['y1'] = rows - y2

	return img, bboxes

def ninety_degree_rotation(img, bboxes, verbose=False):

	rows, cols = img.shape[:2]

	angle = np.random.choice([90,180,270], 1)[0]

	if verbose:
		print('\tRotation (90 degree): ' + str(angle))

	if angle == 270:
		img = np.transpose(img, (1,0,2))
		img = cv2.flip(img, 0)
	elif angle == 180:
		img = cv2.flip(img, -1)
	elif angle == 90:
		img = np.transpose(img, (1,0,2))
		img = cv2.flip(img, 1)

	for bbox in bboxes:
		x1 = bbox['x1']
		x2 = bbox['x2']
		y1 = bbox['y1']
		y2 = bbox['y2']
		if angle == 270:
			bbox['x1'] = y1
			bbox['x2'] = y2
			bbox['y1'] = cols - x2
			bbox['y2'] = cols - x1
		elif angle == 180:
			bbox['x2'] = cols - x1
			bbox['x1'] = cols - x2
			bbox['y2'] = rows - y1
			bbox['y1'] = rows - y2
		elif angle == 90:
			bbox['x1'] = rows - y2
			bbox['x2'] = rows - y1
			bbox['y1'] = x1
			bbox['y2'] = x2

	return img, bboxes

def any_degree_rotation(img, bboxes, verbose=False):

	bboxes_arr = np.array([[bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]for bbox in bboxes])

	height, width = img.shape[:2]

	max_degrees = 3
	angle = np.random.uniform(-1.0 * max_degrees, max_degrees)

	if verbose:
		print('\tRotation (any degree): ' + str(angle))

	cx = width // 2
	cy = height // 2

	# Rotate image.
	mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
	cos = np.abs(mat[0, 0])
	sin = np.abs(mat[0, 1])

	new_width = int((height * sin) + (width * cos))
	new_height = int((height * cos) + (width * sin))

	mat[0, 2] += (new_width / 2) - cx
	mat[1, 2] += (new_height / 2) - cy

	img = cv2.warpAffine(img, mat, (new_width, new_height))

	# Rotate bounding boxes.
	box_widths = (bboxes_arr[:,2] - bboxes_arr[:,0]).reshape(-1,1)
	box_heights = (bboxes_arr[:,3] - bboxes_arr[:,1]).reshape(-1,1)

	x1 = bboxes_arr[:,0].reshape(-1,1)
	y1 = bboxes_arr[:,1].reshape(-1,1)

	x2 = x1 + box_widths
	y2 = y1 

	x3 = x1
	y3 = y1 + box_heights

	x4 = bboxes_arr[:,2].reshape(-1,1)
	y4 = bboxes_arr[:,3].reshape(-1,1)

	corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

	corners = corners.reshape(-1,2)
	corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
	
	corners = np.dot(mat, corners.T).T
	corners = corners.reshape(-1,8)

	x_ = corners[:,[0,2,4,6]]
	y_ = corners[:,[1,3,5,7]]

	xmin = np.min(x_,1).reshape(-1,1)
	ymin = np.min(y_,1).reshape(-1,1)
	xmax = np.max(x_,1).reshape(-1,1)
	ymax = np.max(y_,1).reshape(-1,1)

	bboxes_arr = np.hstack((xmin, ymin, xmax, ymax))

	row_min, row_max, col_min, col_max = strap_img(img)
	img = img[row_min:row_max, col_min:col_max, :]

	bboxes_arr, bboxes_mask = clip_box(bboxes_arr, [col_min, row_min, col_max, row_max], 0.5)
	bboxes = [bboxes[i] for i in range(bboxes_mask.shape[0]) if bboxes_mask[i] == 1]

	for i in range(bboxes_arr.shape[0]):
		bboxes[i]['x1'] = int(bboxes_arr[i, 0]-col_min)
		bboxes[i]['y1'] = int(bboxes_arr[i, 1]-row_min)
		bboxes[i]['x2'] = int(math.ceil(bboxes_arr[i, 2]-col_min))
		bboxes[i]['y2'] = int(math.ceil(bboxes_arr[i, 3]-row_min))

	return img, bboxes

def shear(img, bboxes, verbose=False):

	shear_factor = np.random.uniform(-0.3, 0.3)

	if verbose:
		print('\tShear Mapping: ' + str(shear_factor))

	if shear_factor < 0.0:
		img, bboxes = horizontal_flip(img, bboxes)

	height, width = img.shape[:2]
	bboxes_arr = np.array([[bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]for bbox in bboxes])
	
	mat = np.array(
		[
			[1, abs(shear_factor), 0],
			[0, 1, 0]
		]
	)

	new_width =  width + abs(shear_factor*height)
	bboxes_arr[:,[0,2]] += ((bboxes_arr[:,[1,3]]) * abs(shear_factor) ).astype(int)

	img = cv2.warpAffine(img, mat, (int(new_width), height))

	row_min, row_max, col_min, col_max = strap_img(img)
	img = img[row_min:row_max, col_min:col_max, :]

	for i in range(bboxes_arr.shape[0]):
		bboxes[i]['x1'] = int(bboxes_arr[i, 0]-col_min)
		bboxes[i]['y1'] = int(bboxes_arr[i, 1]-row_min)
		bboxes[i]['x2'] = int(math.ceil(bboxes_arr[i, 2]-col_min))
		bboxes[i]['y2'] = int(math.ceil(bboxes_arr[i, 3]-row_min))

	if shear_factor < 0.0:
		img, bboxes = horizontal_flip(img, bboxes)

	return img, bboxes

def random_crop(img, bboxes, verbose=False):

	if verbose:
		print('\tCropping')

	height, width = img.shape[:2]

	new_width = np.random.randint(int(0.4*width), int(0.8*width))
	new_height = np.random.randint(int(0.4*height), int(0.8*height))

	col_min = np.random.randint(0, width-new_width)
	row_min = np.random.randint(0, height-new_height)

	col_max = col_min + new_width
	row_max = row_min + new_height

	img = img[row_min:row_max, col_min:col_max, :]

	bboxes_arr = np.array([[bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']] for bbox in bboxes])
	bboxes_arr, bboxes_mask = clip_box(bboxes_arr, [col_min, row_min, col_max, row_max], 0.5)
	bboxes = [bboxes[i] for i in range(bboxes_mask.shape[0]) if bboxes_mask[i] == 1]

	for i in range(bboxes_arr.shape[0]):
		bboxes[i]['x1'] = int(bboxes_arr[i, 0]-col_min)
		bboxes[i]['y1'] = int(bboxes_arr[i, 1]-row_min)
		bboxes[i]['x2'] = int(math.ceil(bboxes_arr[i, 2]-col_min))
		bboxes[i]['y2'] = int(math.ceil(bboxes_arr[i, 3]-row_min))
	
	return img, bboxes

def brightness(img, bboxes, verbose=False):

	if verbose:
		print('\tChanging Brightness')

	background_mask = img == 0
	img = img.astype('float32')

	max_brightness = 180
	min_brightness = 75

	avg_brightness = img[~background_mask].mean()

	if avg_brightness <= min_brightness or avg_brightness >= max_brightness:
		img = img.astype('uint8')
		return img, bboxes

	p = (avg_brightness - min_brightness) / (max_brightness - min_brightness)

	if np.random.random() < p: # Make darker.
		img -= np.random.random() * (avg_brightness - min_brightness)

	else: # Make lighter
		img += np.random.random() * (max_brightness - avg_brightness)

	img = np.clip(img, 0, 255)

	img = img.astype('uint8')
	img[background_mask] = 0

	return img, bboxes

def contrast(img, bboxes, verbose=False):

	if verbose:
		print('\tChanging Contrast')

	max_contrast = 180
	min_contrast = 75

	img = skimage.exposure.rescale_intensity(
		img,
		in_range=(
			min_contrast * np.random.random(),
			(255 - max_contrast) * np.random.random() + max_contrast
		)
	)

	return img, bboxes

def salt_and_pepper_noise(img, bboxes, img_type, verbose=False):

	max_amount = 0.3
	min_amount = 0.01
	amount = (max_amount - min_amount) * np.random.random() + min_amount
	salt_vs_pepper = get_truncated_normal(mean=0.5, sd=0.1, low=0, upp=1).rvs(size=1)[0]

	if verbose:
		print('\tAdding Salt and Pepper Noise: ' + str((amount, salt_vs_pepper)))

	if 'grey' in img_type:

		background_mask = img[:, :, 0] == 0

		img_noise = skimage.util.random_noise(
			img[:, :, 0],
			mode='s&p',
			clip=True,
			amount=amount,
			salt_vs_pepper=salt_vs_pepper
		)

		img_noise = skimage.util.img_as_ubyte(img_noise)
		img_noise[background_mask] = 0

		img[:, :, 0] = img_noise
		img[:, :, 1] = img_noise
		img[:, :, 2] = img_noise

	else:

		background_mask = img == 0

		img = skimage.util.random_noise(
			img,
			mode='s&p',
			clip=True,
			amount=amount,
			salt_vs_pepper=salt_vs_pepper
		)

		img = skimage.util.img_as_ubyte(img)
		img[background_mask] = 0

	return img, bboxes

def gaussian_noise(img, bboxes, img_type, verbose=False):

	mean = (0.05 + 0.05) * np.random.random() - 0.05
	var = (0.01 - 0.001) * np.random.random() + 0.001

	if verbose:
		print('\tAdding Gaussian Noise: ' + str((mean, var)))

	if 'grey' in img_type:

		background_mask = img[:, :, 0] == 0

		img_noise = skimage.util.random_noise(
			img[:, :, 0],
			mode='gaussian',
			clip=True,
			mean=mean,
			var=var
		)

		img_noise = skimage.util.img_as_ubyte(img_noise)
		img_noise[background_mask] = 0

		img[:, :, 0] = img_noise
		img[:, :, 1] = img_noise
		img[:, :, 2] = img_noise

	else:

		background_mask = img == 0

		img = skimage.util.random_noise(
			img,
			mode='gaussian',
			clip=True,
			mean=mean,
			var=var
		)

		img = skimage.util.img_as_ubyte(img)
		img[background_mask] = 0

	return img, bboxes

def poisson_noise(img, bboxes, img_type, verbose=False):

	if verbose:
		print('\tAdding Poisson Noise')

	if 'grey' in img_type:

		background_mask = img[:, :, 0] == 0

		img_noise = skimage.util.random_noise(
			img[:, :, 0],
			mode='poisson',
			clip=True
		)

		img_noise = skimage.util.img_as_ubyte(img_noise)
		img_noise[background_mask] = 0

		img[:, :, 0] = img_noise
		img[:, :, 1] = img_noise
		img[:, :, 2] = img_noise

	else:

		background_mask = img == 0

		img = skimage.util.random_noise(
			img,
			mode='poisson',
			clip=True
		)

		img = skimage.util.img_as_ubyte(img)
		img[background_mask] = 0

	return img, bboxes


def augment(img_data, img, config, augment=True, verbose=False):

	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	if augment:

		if verbose:
			print('Performing Augmentation:')
		
		if config.use_horizontal_flips and np.random.random() < 0.5:
			img, img_data_aug['bboxes'] = horizontal_flip(img, img_data_aug['bboxes'], verbose)
		
		if config.use_vertical_flips and np.random.random() < 0.5:
			
			img, img_data_aug['bboxes'] = vertical_flip(img, img_data_aug['bboxes'], verbose)

		if config.use_90_rotations and np.random.random() < 0.5:

			img, img_data_aug['bboxes'] = ninety_degree_rotation(img, img_data_aug['bboxes'], verbose)
		
		if config.use_rotations and np.random.random() < 0.5:
				
			img, img_data_aug['bboxes'] = any_degree_rotation(img, img_data_aug['bboxes'], verbose)
		
		if config.use_shear and np.random.random() < 0.25:

			img, img_data_aug['bboxes'] = shear(img, img_data_aug['bboxes'], verbose)
		
		if config.use_brightness and np.random.random() < 0.5:

			img, img_data_aug['bboxes'] = brightness(img, img_data_aug['bboxes'], verbose)

		if config.use_noise and np.random.random() < 0.5:

			r = np.random.randint(0, 4)
			if r == 0:
				img, img_data_aug['bboxes'] = salt_and_pepper_noise(img, img_data_aug['bboxes'], config.img_types[0], verbose)
			elif r == 1:
				img, img_data_aug['bboxes'] = gaussian_noise(img, img_data_aug['bboxes'], config.img_types[0], verbose)
			elif r == 2:
				img, img_data_aug['bboxes'] = poisson_noise(img, img_data_aug['bboxes'], config.img_types[0], verbose)
			elif r == 3:
				img, img_data_aug['bboxes'] = contrast(img, img_data_aug['bboxes'], verbose)
		
		img_data_aug['width'] = img.shape[1]
		img_data_aug['height'] = img.shape[0]
	
	return img_data_aug, img
