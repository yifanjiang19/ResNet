import numpy as np
import os
import cPickle as pickle

data_dir = "cifar-100-python"
train_data = os.path.join(data_dir,"train")
test_data = os.path.join(data_dir,"test")
class CIFAR(object):

	def __init__(self,batch_size):

		self.index = 0
		self.batch_size = batch_size

	def open_file(self,data):
		f = open(data,'rb')
		dict = pickle.load(f)
		f.close()
		return dict

	def one_hot(self,x):
		a = np.zeros(100)
                a[x] = 1
		return a


	def input_train(self):
                train_dict = self.open_file(train_data)
		x_train = train_dict['data']/np.float32(255)
		y_train = train_dict['fine_labels']
		print(x_train[1])
		return x_train,y_train

	def input_test(self):
		test_dict = self.open_file(test_data)
		x_test = test_dict['data']
		y_test = test_dict['fine_labels']
		return x_test,y_test
