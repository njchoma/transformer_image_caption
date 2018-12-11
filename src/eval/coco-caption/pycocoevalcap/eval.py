__author__ = 'tylin'
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from spice.spice import Spice

import pickle
import json

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self, gt_path, results_path):
        #imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        gt_pkl = pickle.load(open(gt_path, 'rb'))
        for key in gt_pkl.keys():
            sample = gt_pkl[key]
            img_id = int(sample['image_id'])
            caption = sample['caption']
            if img_id in gts:
                gts[img_id].append({'image_id': img_id, 'caption': caption})
            else:
                gts[img_id] = [{'image_id': img_id, 'caption': caption}]
        #print(gts)

        res = {}        
        res_json = json.load(open(results_path,'rb'))
        for sample in res_json:
            img_id = int(sample['image_id'])
            res[img_id] = [sample]
        #print(res)
        #print(1/0)
        #for imgId in imgIds:
            #gts[imgId] = self.coco.imgToAnns[imgId]
            #print(gts[imgId])
            #res[imgId] = self.cocoRes.imgToAnns[imgId]
            #print(imgId)
            #print(res[imgId])
            #print(1/0)
        #print(res)
        #print(1/0)
        # =================================================
        # Set up scorers
        # =================================================
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
            #(Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print "%s: %0.3f"%(method, score)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
