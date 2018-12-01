from pycocotools.coco import COCO
import os
import pickle
import cPickle
#from easydict import EasyDict as edict

data_root = '/scratch/ovd208/COCO_features/data'
data_type = 'train'
annotation_path = os.path.join(data_root, "karpathy_splits",  data_type + "_karpathy.pkl")
train_ks_anns = pickle.load(open(annotation_path,"rb"))
mini_data_size = 100
medium_data_size = 20000

mini_captions = {}
medium_captions = {}

imgid2idx = pickle.load(open(os.path.join(data_root, "karpathy_splits", data_type + "_ks_imgid2idx.pkl"), 'rb'))

counter=0
for key in train_ks_anns:
    caption = train_ks_anns[key]['caption']
    image_id = train_ks_anns[key]['image_id']

    if imgid2idx[image_id] < mini_data_size:
        mini_captions[counter] = {'caption': caption, 'image_id': image_id}
    if imgid2idx[image_id] < medium_data_size:
        medium_captions[counter] = {'caption': caption, 'image_id': image_id}
    counter = counter + 1

print('Size of mini data: ' + str(len(mini_captions)))
mini_anno_file_name = os.path.join(data_root, "karpathy_splits", data_type + '_karpathy_mini.pkl')
mini_anno_file = open(mini_anno_file_name, 'wb')
pickle.dump(mini_captions, mini_anno_file)
mini_anno_file.close()

print('Size of medium data: ' + str(len(medium_captions)))
med_anno_file_name = os.path.join(data_root, "karpathy_splits", data_type + '_karpathy_medium.pkl')
med_anno_file = open(med_anno_file_name, 'wb')
pickle.dump(medium_captions, med_anno_file)
med_anno_file.close()
