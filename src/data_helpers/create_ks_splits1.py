from pycocotools.coco import COCO
import os
import pickle
import cPickle
#from easydict import EasyDict as edict

data_root = '/scratch/ovd208/COCO_features/data'
data_types = ['train', 'val']
annotation_paths = [os.path.join(data_root, "captions", "annotations", "captions_" + d_type + "2014.json") for d_type in data_types]
cocos = [COCO(ann_path) for ann_path in annotation_paths]
train_val_ids = [list(coco.anns.keys()) for coco in cocos]

#print("Number of COCO train + val ids: " + len(ids))

#mini_dict = edict()
#mini_captions = {}
splits = ['train', 'val', 'test']

#imgid2idx = cPickle.load(open(os.path.join(data_root, data_type + "36_imgid2idx.pkl"), 'rb'))
captions={}
split_ids = {}
for split in splits:
    split_ids[split] = []
    captions[split] = {}
    for line in open(os.path.join(data_root, "karpathy_splits", "karpathy_" + split + "_images.txt"), "r"):
        img_id = line.split(" ")[1]
        split_ids[split].append(int(img_id))

for counter,ids in enumerate(train_val_ids):
    for c1, i in enumerate(ids):
        if c1%100==0:
            print(str(c1) + "\n")
        coco = cocos[counter]
        caption = coco.anns[i]['caption']
        image_id = coco.anns[i]['image_id']
        #print(type(image_id))
        #print(type(split_ids['train'][0]))
        if image_id in split_ids['train']:
            captions['train'][i] = {'caption': caption, 'image_id': image_id}
        elif image_id in split_ids['val']:
            captions['val'][i] = {'caption': caption, 'image_id': image_id}
        else:
            captions['test'][i] = {'caption': caption, 'image_id': image_id}
#mini_dict.anns1 = mini_captions

split_file_names = [os.path.join(data_root, "karpathy_splits", split + '_karpathy.pkl') for split in splits]
split_anno_files = [open(file_name, 'wb') for file_name in split_file_names]
for i,split in enumerate(splits):
    print("saving " + split + ", len: " + str(len(captions[split])))
    pickle.dump(captions[split], split_anno_files[i])
    split_anno_files[i].close()
