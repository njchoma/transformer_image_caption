import os
import h5py
import numpy as np
import pickle
splits = ['train', 'val', 'test']
data_root = '/scratch/ovd208/COCO_features/data'

train_file_name = os.path.join(data_root, 'train36.hdf5')
train_data_h5 = h5py.File(train_file_name, 'r')
train_features = np.array(train_data_h5.get('image_features'))


val_file_name = os.path.join(data_root, 'val36.hdf5')
val_data_h5 = h5py.File(val_file_name, 'r')
val_features = np.array(val_data_h5.get('image_features'))

train_imgid2idx = pickle.load(open(os.path.join(data_root, 'train36_imgid2idx.pkl'), "rb"))
val_imgid2idx = pickle.load(open(os.path.join(data_root, 'val36_imgid2idx.pkl'), "rb"))

train_ks_imgid2idx = {}
val_ks_imgid2idx = {}
test_ks_imgid2idx = {}

for split in splits:
    print('Doing: ' + split)
    file_name = os.path.join(data_root,'karpathy_splits', split + '36_ks.hdf5')
    data = h5py.File(file_name, "w")
    
    f = open(os.path.join(data_root, 'karpathy_splits', 'karpathy_' + split + '_images.txt'),'r')
    f_lines = f.readlines()
    data_size = len(f_lines)
    print("No of images for this split: " + str(data_size))
    num_fixed_boxes = 36
    feature_length = 2048

    dataset = data.create_dataset('image_features', (data_size, num_fixed_boxes, feature_length), 'f')
    counter=0

    mini_data = None
    mini_dataset = None
    med_data = None
    med_dataset = None
    #images
    mini_size = 100
    medium_size = 20000
    #make mini and medium datasets for trian split
    if split =='train':
        mini_file_name = os.path.join(data_root, 'karpathy_splits', 'train36_ks_mini.hdf5')
        med_file_name = os.path.join(data_root, 'karpathy_splits', 'train36_ks_med.hdf5')
        mini_data = h5py.File(mini_file_name, "w")
        med_data = h5py.File(med_file_name, "w")
        mini_dataset = mini_data.create_dataset('image_features', (mini_size, num_fixed_boxes, feature_length), 'f')
        med_dataset = med_data.create_dataset('image_features', (medium_size, num_fixed_boxes, feature_length), 'f')

    for ii,line in enumerate(f_lines):
        if ii%1000==0:
            print(str(ii) + "\n")
        image_id = line.split(" ")[1]
        image_id = int(image_id)
        idx=-1
        
        if image_id in train_imgid2idx.keys():
            idx = train_imgid2idx[image_id]
            dataset[counter,:,:] = train_features[idx,:,:]
            if counter < mini_size and split == 'train':
                mini_dataset[counter,:,:] = train_features[idx,:,:]
            if counter < medium_size and split == 'train':
                med_dataset[counter, :, :] = train_features[idx,:,:]
        elif image_id in val_imgid2idx.keys():
            idx = val_imgid2idx[image_id]      
            dataset[counter,:,:] = val_features[idx,:,:]
            if counter < mini_size and split == 'train':
                mini_dataset[counter,:,:] = val_features[idx,:,:]
            if counter < medium_size and split == 'train':
                med_dataset[counter, :, :] = val_features[idx,:,:]
        else:
            raise ValueError('img id faulty?')   
        if split=='train':
            train_ks_imgid2idx[image_id] = counter
        elif split=='val':
            val_ks_imgid2idx[image_id] = counter
        else:
            test_ks_imgid2idx[image_id] = counter
        counter+=1

    data.close()
    
    if split=='train':
        mini_data.close()
        med_data.close()
        pickle.dump(train_ks_imgid2idx, open(os.path.join(data_root, 'karpathy_splits', 'train_ks_imgid2idx.pkl'), 'wb')) 
    elif split=='val':
        pickle.dump(val_ks_imgid2idx, open(os.path.join(data_root, 'karpathy_splits', 'val_ks_imgid2idx.pkl'), 'wb'))
    else:
        pickle.dump(test_ks_imgid2idx, open(os.path.join(data_root, 'karpathy_splits', 'test_ks_imgid2idx.pkl'), 'wb'))
