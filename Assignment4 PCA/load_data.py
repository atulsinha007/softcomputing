import cv2
import os
import numpy as np

TRAIN_DIR = '/home/ak/Downloads/SEM6/Sem6 Assignments/SCO/Assignment4 PCA/data_set/train_set'
TEST_DIR = '/home/ak/Downloads/SEM6/Sem6 Assignments/SCO/Assignment4 PCA/data_set/test_set'

train_dir = '/home/ak/Downloads/SEM6/Sem6 Assignments/SCO/Assignment4 PCA/Generated/train_set'
test_dir = '/home/ak/Downloads/SEM6/Sem6 Assignments/SCO/Assignment4 PCA/Generated/test_set'


def create_dataset(fileList, domain, dir_name):

	for i, file in enumerate(fileList):
		oldfile = file
		file = os.path.join(dir_name, file)
		img = cv2.imread(file)
		resized_img = cv2.resize(img, (312,416))
		resized_img = cv2.cvtColor( resized_img, cv2.COLOR_RGB2GRAY )
		cv2.imwrite('/home/ak/Downloads/SEM6/Sem6 Assignments/SCO/Assignment4 PCA/Generated/' + domain + '/'+ oldfile + '_resized_bw',resized_img)
		

def create():	

	fileList = sorted(os.listdir(TRAIN_DIR))
	create_dataset(fileList, "train_set", TRAIN_DIR)
	fileList = sorted(os.listdir(TEST_DIR))
	create_dataset(fileList, "test_set", TEST_DIR)


def read_data(fileList, dir_name):
	
	x = []
	y = []
	for i, file in enumerate(fileList):
		oldfile = file
		file = os.path.join(dir_name, file)
		img = cv2.imread(file)
		data_reshaped = img.flatten()
		y.append(int(oldfile[0]))
		x.append(data_reshaped)
	y = np.asarray(y)
	y = y.reshape(-1, 1)
	x = np.asarray(x)
	x = x.transpose()
	return x, y



def load_data():

	fileList = sorted(os.listdir(train_dir))
	trainx, trainy = read_data(fileList, train_dir)

	fileList = sorted(os.listdir(test_dir))
	testx, testy = read_data(fileList, test_dir)

	return trainx, trainy, testx, testy


def main():
	create()

if __name__ == '__main__':
	main()