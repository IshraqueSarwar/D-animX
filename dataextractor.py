import cv2
import pickle
import glob
import os
from tqdm import tqdm


RESET_DATAFILE = True


# the raw sprite sheets
pre_process_files= 'data/preprocessed/*.png'
# file that contains important counters e.g the current ID and ID list
datfile = 'data/dat.pickle'
# the folder where the output/processed images will be saved
processed_path = 'data/processed/'
# boolean that tells if the spritesheet is already cropped or not
cropped = False

# the default dimentions of each image
dim = 32

# TO DECIDE: (function implementation)def read_write_datfile(read = None, write = None, current_ID = None, ID_list = None):





# read in the Current id and id list if present else, create one.
if os.path.exists(datfile) and not RESET_DATAFILE:
	with open(datfile, 'rb') as f:
		saved_vars = pickle.load(f)
		current_ID = saved_vars[0]
		ID_list = saved_vars[1]
else:
	# as the files doesn't exist we need to create the current ID and ID list
	current_ID = 0
	ID_list = []












# if already cropped then we just read serially or sort them to serial
# and then label them using id and serial number
if cropped:
	name_to_ID = {}
	name_to_index = []
	# reading all the png files in /data/preprocessed/
	for imgfile in glob.glob(pre_process_files):
		# removing the file extensions to get the names only. Currently supports removing
		# jpeg, jpg and png files.
		file_name = imgfile.replace('.png','').replace('jpeg','').replace('.jpg','')
		
		# checking if the file name is unique, if so then append
		# [:-1] removes the serial number from the file_name
		if file_name[:-1] not in name_to_index:
			name_to_index.append(file_name[:-1])
			name_to_ID[file_name[:-1]] = 0

		
		img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)

		cv2.imwrite(f'{processed_path}{name_to_index.index(file_name[:-1]) +current_ID}_{name_to_ID[file_name[:-1]]}.png', img)
		name_to_ID[file_name[:-1]]+=1
	current_ID+=len(name_to_index)

	print(name_to_index)






# if not cropped then we take in each sprite sheet and crop it in to
# single image. each row will have unique id and the column represents
# the serial number
else:
	# reading all the png files in /data/preprocessed/ 
	for imgfile in glob.glob(pre_process_files):
		# img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
		img = cv2.imread(imgfile)
		# iterating over the number of rows
		for i in tqdm(range(1, int(img.shape[0]/dim)+1)):
			# determing the end pixel vertically
			height = dim*i

			#iterating over the number of columns
			for j in range(1, int(img.shape[1]/dim)+1):
				# determing the end pixel horizontally
				width = dim*j

				# using the width, height calculated and the dim, the image is cropped
				cropped_img = img[height-dim:height, width-dim:width]

				# save the cropped image using the current_ID 
				# and j (which also acts as the serial number of anim progress)
				cv2.imwrite(f'{processed_path}{current_ID}_{j}.png', cropped_img)


			# adding current_ID to the list of ids
			ID_list.append(current_ID)
			# updating the current ID
			current_ID+=1





# write variables to save to datafile
with open(datfile, 'wb') as f:
	save_var = [current_ID, ID_list]
	pickle.dump(save_var, f)
