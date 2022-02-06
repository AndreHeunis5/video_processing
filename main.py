import argparse
import os

import cv2

from masking.ContourSegmenter import ContourSegmenter

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Segment produce on a scale.')
	parser.add_argument(
		'--input_path',
		type=str,
		default='video_1.mp4',
		help='Path to mp4 video to segment.')
	parser.add_argument(
		'--output_path',
		type=str,
		default='',
		help='Path to mp4 video to segment.')
	parser.add_argument(
		'--method',
		default='contour',
		help="Only 'contour' supported")

	args = parser.parse_args()

	input_path = args.input_path
	if input_path[-4:] != '.mp4':
		raise ValueError('Need to specify an mp4 input file.')

	output_path = args.output_path
	if output_path == '':
		output_path = input_path[:-4] + '_processed.mp4'

	# Check if output path exists and create it if it doesn't
	path_to_outfile = os.path.join(*output_path.split('/')[:-1])
	if not os.path.exists(path_to_outfile):
		os.makedirs(path_to_outfile)

	print('Using input at ', input_path)
	print('Output will be saved to ', output_path)

	method = args.method
	if method == 'contour':
		segmenter = ContourSegmenter()
	else:
		raise ValueError(f'Unsupported method {method} chosen.')

	# Load the input video
	raw_video = cv2.VideoCapture(input_path)
	fps = raw_video.get(cv2.CAP_PROP_FPS)
	width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Create storage for the output video
	out_video = cv2.VideoWriter(
		output_path,
		apiPreference=0,
		fourcc=cv2.VideoWriter_fourcc(*'MP4V'),
		fps=fps,
		frameSize=(width, height))

	# Segment each frame of the input and store in the output
	frame_ind = 0
	while raw_video.isOpened():
		ret, frame = raw_video.read()
		if not ret:
			break

		res = segmenter.segment_frame(frame)
		out_video.write(res)

		if frame_ind % 100 == 0:
			print('Frames processed: ', frame_ind)
		frame_ind += 1
	raw_video.release()
	out_video.release()
