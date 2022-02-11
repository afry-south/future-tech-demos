import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random

random.seed(1143)

def populate_train_list(lowlight_images_path: str) -> list[str]:
	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	train_list = image_list_lowlight
	random.shuffle(train_list)

	return train_list

class lowlight_loader(data.Dataset):
	def __init__(self, lowlight_images_path: str, size: int = 256):
		self.data_list = populate_train_list(lowlight_images_path) 
		self.size = size

		print(f"Total training examples: {len(self)}")

	def __getitem__(self, index):
		data_lowlight_path = self.data_list[index]
		data_lowlight = Image.open(data_lowlight_path)
		
		data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)
		data_lowlight = (np.asarray(data_lowlight) / 255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return data_lowlight.permute(2, 0, 1)

	def __len__(self):
		return len(self.data_list)