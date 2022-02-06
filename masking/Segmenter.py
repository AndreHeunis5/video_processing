from abc import ABC, abstractmethod


class Segmenter(ABC):

	@abstractmethod
	def segment_frame(self, frame):
		pass
