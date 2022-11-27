import cv2
import os
import glob
from tqdm import tqdm
import pickle


class DataProcessor:
	def __init__(self, dim_size = 32, preprocessed_folder = 'data/preprocessed', processed_folder = 'data/processed'):
		self.datafile = 'data/dat.pickle'
		self.preprocessed_folder = preprocessed_folder
		self.processed_folder = processed_folder
		self.DIM_SIZE = dim_size
		self.pre_cropped_filenames = []
		self.pre_non_cropped_filenames = []
		self.current_ID, self.ID_list = self.initialize_hyperparams(RESET_DATAFILE=False)


	def save_hyperparams(self):
		with open(self.datafile, 'wb') as f:
			save_var = [self.current_ID, self.ID_list]
			pickle.dump(save_var, f)

	def initialize_hyperparams(self, RESET_DATAFILE):
		if os.path.exists(self.datafile) and not RESET_DATAFILE:
			with open(self.datafile, 'rb') as f:
				saved_vars = pickle.load(f)
				current_ID = saved_vars[0]
				ID_list = saved_vars[1]
		else:
			current_ID = 0
			ID_list = [] 

		return current_ID, ID_list


	def process(self):
		for imgfile in glob.glob(f"{self.preprocessed_folder}/*.png"):
			img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
			if img.shape[0] == self.DIM_SIZE:
				self.pre_cropped_filenames.append(imgfile)
			else:
				self.pre_non_cropped_filenames.append(imgfile)
		
		self.process_already_cropped_files()
		self.process_non_cropped_files()
		self.save_hyperparams()


	def process_already_cropped_files(self):
		name_to_ID = {}
		name_to_index = []
		for imgfile in tqdm(self.pre_cropped_filenames):
			file_name = imgfile.replace('.png','').replace('jpeg','').replace('.jpg','')
			if file_name[:-1] not in name_to_index:
				name_to_index.append(file_name[:-1])
				name_to_ID[file_name[:-1]] = 0

			img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)

			cv2.imwrite(f'{self.processed_folder}/{name_to_index.index(file_name[:-1]) +self.current_ID}_{name_to_ID[file_name[:-1]]}.png', img)
			name_to_ID[file_name[:-1]]+=1

			if self.current_ID not in self.ID_list:
				self.ID_list.append(self.current_ID)

		self.current_ID+=len(name_to_index)


	def process_non_cropped_files(self):
		for imgfile in tqdm(self.pre_non_cropped_filenames):
			img = cv2.imread(imgfile, cv2.IMREAD_UNCHANGED)

			for x in range(1, int(img.shape[0]/self.DIM_SIZE)+1):
				height = self.DIM_SIZE*x
				for y in range(1, int(img.shape[1]/self.DIM_SIZE)+1):
					width = self.DIM_SIZE*y

					cropped_img = img[height-self.DIM_SIZE:height, width-self.DIM_SIZE:width]
					cv2.imwrite(f'{self.processed_folder}/{self.current_ID}_{y}.png', cropped_img)


				self.ID_list.append(self.current_ID)
				self.current_ID+=1


	

DP = DataProcessor()
DP.process()

