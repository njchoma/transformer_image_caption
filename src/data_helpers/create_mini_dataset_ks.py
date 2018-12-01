import os
import h5py
import numpy as np
data_type = 'train'
data_root = '/scratch/ovd208/COCO_features/data/'
file_name = os.path.join(data_root, "karpathy_splits", data_type + '36_ks.hdf5')
data_h5 = h5py.File(file_name, 'r')

print('File loaded!')
train_features = np.array(data_h5.get('image_features'))

mini_file_name = os.path.join(data_root, "karpathy_splits", data_type + '36_ks_mini.hdf5')
mini_data = h5py.File(mini_file_name, "w")
mini_data_size = 500

med_file_name = os.path.join(data_root, "karpathy_splits", data_type + '36_ks_medium.hdf5')
med_data = h5py.File(med_file_name, "w")
med_data_size = 100000
num_fixed_boxes = 36
feature_length = 2048

mini_dataset = mini_data.create_dataset('image_features', (mini_data_size, num_fixed_boxes, feature_length), 'f')
med_dataset = med_data.create_dataset('image_features', (med_data_size, num_fixed_boxes, feature_length), 'f')

for i in range(mini_data_size):
    mini_dataset[i,:,:] = train_features[i,:,:]
mini_data.close()

print('Wrote the mini dataset')

for i in range(med_data_size):
    med_dataset[i,:,:] = train_features[i,:,:]
med_data.close()

print('Wrote the medium dataset')
