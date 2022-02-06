import numpy as np
import cv2

from masking.Segmenter import Segmenter


class ContourSegmenter(Segmenter):
	"""
	Attempt to do foreground / background segmentation with a random edge / contour detection method found on the
	internet. See docstring in segment_frame() method for details of the algorithm.

	NOTE: Call reset_base_frame() before processing a new video.

	Attributes:
		background_subtraction_threshold:	Threshold for pixel difference between the base frame and the current frame.
		canny_low:												Canny edge detection low threshold.
		canny_high:												Canny edge detection high threshold.
		blur															Used for softening edges / contours.
		mask_dilate_iter									Number of iterations for dilation on the mask.
		mask_erode_iter										Number of iterations for erosion on the mask.
		min_contour_area_proportion:			Minimum area of a contour relative to the frame size.
		max_contour_area_proportion:			Maximum area of a contour relative to the frame size.
		frame_base:												The base image for a video.
	"""

	def __init__(self):
		self.background_subtraction_threshold = 30
		self.canny_low = 15
		self.canny_high = 150
		self.blur = 21
		self.mask_dilate_iter = 5
		self.mask_erode_iter = 5
		self.min_contour_area_proportion = 0.0001
		self.max_contour_area_proportion = 0.95
		self.frame_base = None

	def segment_frame(self, frame: np.array) -> np.array:
		"""
		Segment a single frame as follows:
			1. Assume the first frame contains the pure background that always needs to be removed. Save it.
			2. For each frame:
				2.1	Subtract the base background
				2.2 Now follow the method detailed here
					  https://towardsdatascience.com/background-removal-with-python-b61671d1508a. The first step is to detect
					  contours in the frame.
				2.3 Apply any contours within the accepted size range to the background mask.
				2.4 Apply the background mask to the frame by colouring areas outside the contours black.

		:param frame:	3 channel image to segment. Values between 0 and 255.
		:return:			3 channel image with the supposed background set to black.
		"""
		masked_frame = self.__subtract_base_background(frame)

		# Edge / contour detection
		image_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(image_gray, self.canny_low, self.canny_high)
		edges = cv2.dilate(edges, None)
		edges = cv2.erode(edges, None)
		contour_info = [(c, cv2.contourArea(c),) for c in
										cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]

		# calculate max and min contour areas in terms of pixels
		image_area = frame.shape[0] * frame.shape[1]
		max_contour_area = self.max_contour_area_proportion * image_area
		min_contour_area = self.min_contour_area_proportion * image_area

		# Go through and find relevant contours and apply to mask
		mask = np.zeros(edges.shape, dtype=np.uint8)
		for contour in contour_info:
			if contour[1] > min_contour_area and contour[1] < max_contour_area:
				mask = cv2.fillConvexPoly(mask, contour[0], 255)

		mask = cv2.dilate(mask, None, iterations=self.mask_dilate_iter)
		mask = cv2.erode(mask, None, iterations=self.mask_erode_iter)
		mask = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)

		mask_stack = np.dstack([mask] * 3)
		mask_stack = mask_stack.astype('float32') / 255.0
		masked_frame = masked_frame.astype('float32') / 255.0

		masked_background = (mask_stack * masked_frame) + ((1-mask_stack) * (0.0, 0.0, 0.0))
		masked_background = (masked_background * 255).astype('uint8')

		return masked_background

	def __subtract_base_background(self, frame: np.array) -> np.array:
		"""
		Assume the first frame in a video only contains the background and remove it from all frames.

		:param frame:	3 channel image to segment. Values between 0 and 255.
		:return:			Frame with the base image removed.
		"""
		if self.frame_base is None:
			self.frame_base = np.copy(frame)

		mask = cv2.absdiff(
			cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
			cv2.cvtColor(self.frame_base, cv2.COLOR_BGR2GRAY))

		imask = mask > self.background_subtraction_threshold
		masked_frame = np.zeros_like(frame, np.uint8)
		masked_frame[imask] = frame[imask]

		return masked_frame

	def reset_base_frame(self):
		""" Reset the base frame before processing a new video. """
		self.frame_base = None
