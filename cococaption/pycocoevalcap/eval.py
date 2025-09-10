from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
#from .spice.spice import Spice
from .wmd.wmd import WMD
import ipdb

class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalVideos = []
        self.eval = {}
        self.videoToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        #self.params = {'video_id': coco.getVideoIds()}
        if cocoRes is not None:
            self.params = {'video_id': cocoRes.getVideoIds()}
        else:
            self.params = {'video_id': []}

        #self.Spice = Spice()

    def evaluate(self):
        videoIds = self.params['video_id']
        # videoIds = self.coco.getvideoIds()
        gts = {}
        res = {}
        print('total test num:{}'.format(len(videoIds)))
        for videoId in videoIds:
            gts[videoId] = self.coco.videoToAnns[videoId]
            res[videoId] = self.cocoRes.videoToAnns[videoId]

        #ipdb.set_trace()
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        # import ipdb 
        # ipdb.set_trace()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(self.Spice, "SPICE"),
            #(WMD(),   "WMD"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setvideoToEvalvideos(scs, list(gts.keys()), m)
                    #print("%s: %0.1f"%(m, sc*100))
            else:
                self.setEval(score, method)
                self.setvideoToEvalvideos(scores, list(gts.keys()), method)
                #print("%s: %0.1f"%(method, score*100))
        self.setEvalvideos()

    def evaluate_cider(self):
        videoIds_from_params = self.params['video_id']
        videoIds_generated = [f"video{i}_{0}" for i in range(7010)]
        videoIds = list(set(videoIds_from_params) | set(videoIds_generated))
        # print(videoIds)
        videoIds = sorted(
    videoIds,
    key=lambda x: (
        int(x.replace("video", "").split("_")[0]),  # numéro principal
        int(x.split("_")[1])                        # suffixe (ex: 0, 1, 2...)
    )
)
        print(videoIds)
        # videoIds = self.coco.getvideoIds()
        gts = {}
        res = {}
        print('total test num:{}'.format(len(videoIds)))
        for videoId in videoIds:
        # Ground truth
            if videoId in self.coco.videoToAnns:
                gts[videoId] = self.coco.videoToAnns[videoId]
            else:
                gts[videoId] = [{'caption': 'placeholder'}]  

            if videoId in self.cocoRes.videoToAnns:
                res[videoId] = self.cocoRes.videoToAnns[videoId]
            else:
                res[videoId] = [{'caption': ''}]  

        #ipdb.set_trace()
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        # import ipdb 
        # ipdb.set_trace()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Cider(), "CIDEr"),
            #(self.Spice, "SPICE"),
            #(WMD(),   "WMD"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setvideoToEvalvideos(scs, list(gts.keys()), m)
                    #print("%s: %0.1f"%(m, sc*100))
            else:
                print(score)
                self.setEval(scores, method)
                self.setvideoToEvalvideos(scores, list(gts.keys()), method)
                #print("%s: %0.1f"%(method, score*100))
        self.setEvalvideos()

    def setEval(self, score, method):
        self.eval[method] = score

    def setvideoToEvalvideos(self, scores, videoIds, method):
        for videoId, score in zip(videoIds, scores):
            if not videoId in self.videoToEval:
                self.videoToEval[videoId] = {}
                self.videoToEval[videoId]["video_id"] = videoId
            self.videoToEval[videoId][method] = score

    def setEvalvideos(self):
        self.evalvideos = [eval for videoId, eval in list(self.videoToEval.items())]
