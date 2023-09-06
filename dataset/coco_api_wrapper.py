# These are just some wrappers of coco api (in pycocotools).
# SingletonCOCOFactory prevents creating multiple COCO instances in a same process.
# SilentCOCO prevents printing messages.
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py

from pycocotools.coco import COCO
import json
from collections import defaultdict


class SingletonCOCOFactory():
    '''
    Singleton factory for COCO class.
    Stores at most one SilentCOCO object for each annotation file name.
    Note that this class cannot prevent creating multiple COCO instances across different processes.
    '''
    coco_dict = {}
    @classmethod
    def get_coco(cls, annotation_file, force_get_new=False):
        if force_get_new:
            # Create new
            return SilentCOCO(annotation_file)
        
        if annotation_file in cls.coco_dict:
            # Recycle
            return cls.coco_dict[annotation_file]

        else:
            # Create new
            cls.coco_dict[annotation_file] = SilentCOCO(annotation_file)
            return cls.coco_dict[annotation_file]


class SilentCOCO(COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats