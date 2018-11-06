from pycocotools.coco import COCO
import os
import pickle
import cPickle
#from easydict import EasyDict as edict

data_root = '/scratch/ovd208/COCO_features/data'
data_type = 'train'
annotation_path = os.path.join(data_root, "captions", "annotations", "captions_" + data_type + "2014.json")
coco = COCO(annotation_path)

mini_data_size = 100
ids = list(coco.anns.keys())

#mini_dict = edict()
mini_captions = {}

imgid2idx = cPickle.load(open(os.path.join(data_root, data_type + "36_imgid2idx.pkl"), 'rb'))

for i in ids:
    caption = coco.anns[i]['caption']
    image_id = coco.anns[i]['image_id']

    if imgid2idx[image_id] < mini_data_size:
        mini_captions[i] = {'caption': caption, 'image_id': image_id}
    
#mini_dict.anns1 = mini_captions

mini_anno_file_name = os.path.join(data_root, 'mini_anno.pkl')
mini_anno_file = open(mini_anno_file_name, 'wb')
pickle.dump(mini_captions, mini_anno_file)
mini_anno_file.close()
