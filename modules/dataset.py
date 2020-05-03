from torch.utils.data import Dataset
import torch
import random

class CropDataset(Dataset):

	def __init__(self, dataframe, bands=['B01', 'B02'], crop_size=3, classf=lambda a: a - 1, need_id=False, autobalance=False, balance_proba=0.5):
		self.df = dataframe
		self.crop_size = crop_size
		self.bands = bands
		self.classf = classf
		self.need_id = need_id
		self.autobalance = autobalance
		self.balance_proba = balance_proba

		if self.autobalance:
			self.balance()

	def get_bands(self, df, idx, bands, crop_size):
		img = []
		for i, band in enumerate(bands):
			img.append(torch.Tensor(df.loc[idx, band].reshape((self.crop_size, self.crop_size))))
		img = torch.stack(img, axis=0)
		return img.unsqueeze(1)

	def balance(self):
		self.target_df = self.df[self.df['class'] == self.autobalance].reset_index()
		self.not_target_df = self.df[self.df['class'] != self.autobalance].reset_index()


	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):

		if self.autobalance:
			if random.random() < self.balance_proba:
				df = self.target_df
			else:
				df = self.not_target_df

			#Resizing index
			idx = int((idx / len(self.df)) * len(df))

		else:
			df = self.df

		if self.need_id:
			return self.get_bands(df, idx, self.bands, self.crop_size), df.loc[idx, 'field_id']
		return self.get_bands(df, idx, self.bands, self.crop_size), self.classf(df.loc[idx, 'class'])
