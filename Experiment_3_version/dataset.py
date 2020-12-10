import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import itertools as IT
import logging
import perception

class CLEVRGroundingDataset(Dataset):
    """CLEVR scenes transformed into grounding representations through a perception module"""

    def __init__(self, total_imgs=10, group_size=1, csv_file='../scenes_train.json'):
        with open(csv_file) as f:
            scenes_json = json.load(f)
            scenes_json = scenes_json['scenes']
            f.close()
        
        self.num_groups=int(total_imgs/group_size)
        if total_imgs%group_size != 0: logging.getLogger(__name__).warning("group size error, will only use %f images" % (self.num_groups*group_size))

        # remove scenes with missing masks
        scenes_json = [sc for sc in scenes_json if len(sc['objects']) == len(sc['objects_detection'])]


        random.seed(7)
        split_scenes = list(grouper(group_size, scenes_json))
        scenes_subset = random.sample(split_scenes, self.num_groups)

        #possible features of an object (excluding 3 pixel-coords)
        obj_colors = ['gray','blue','brown','yellow','red','green','purple','cyan']
        obj_sizes = ['small','large']
        obj_shapes = ['cube','sphere','cylinder']
        obj_materials = ['rubber','metal']
        ## obj_feat : ['gray','blue','brown','yellow','red','green','purple','cyan','small','large','cube','sphere','cylinder','rubber','metal']
        obj_feat = obj_colors + obj_sizes + obj_shapes + obj_materials
        obj_directions = ['right','left','front','behind']

        self.obj_data = []
        self.obj_attr = []
        self.obj_not_attr = []
        self.pairs = []
        for b, sc_batch in enumerate(scenes_subset):
            full_obj_set = [] #list of all objects from all scenes in group
            full_obj_feat = [] #list of all features per object in group
            rel_pairs = {'right': [], 'left':[], 'front':[], 'behind':[]}
            obj_attr, not_obj_attr = {}, {}

            for idx, scene in enumerate(sc_batch):

                # Number of objects parsed from data (used for indexing of object relations)
                num_obj = len(full_obj_set) 

                # build set of objects in an image
                for ido, o in enumerate(scene['objects']):
                    color_vec = [(o['color'] == c)*1 for c in obj_colors]
                    size_vec = [(o['size'] == s)*1 for s in obj_sizes]
                    shape_vec = [(o['shape'] == sh)*1 for sh in obj_shapes]
                    material_vec =[(o['material'] == m)*1 for m in obj_materials]
                    #pixel_vec = [o['pixel_coords'][0]/480, o['pixel_coords'][1]/320, o['pixel_coords'][0]/32]
                    # object features (used to categories and collect groups of same-featured objects)
                    full_obj_feat.append(color_vec + size_vec + shape_vec + material_vec)
                    # set of objects in current scene, and of all objects witnessed
                    #scene_obj_set.append(perception.get_vector(scene,idx))
                    full_obj_set.append(perception.get_vector(scene,ido, mode='val'))
                #print('All objects: ', obj_set)

                for rel in obj_directions:
                    for i in range(len(scene['objects'])):
                        for j in range(len(scene['relationships'][rel][i])):
                            r_pair = [num_obj+i, num_obj+scene['relationships'][rel][i][j]]    
                            rel_pairs[rel].append(r_pair)

            ### Create Subsets of object attributes
            for i, feat in enumerate(obj_feat):
                obj_attr[feat] = [x for (idx, x) in enumerate(full_obj_set) if full_obj_feat[idx][i]==1]
                not_obj_attr[feat] = [x for (idx, x) in enumerate(full_obj_set) if full_obj_feat[idx][i]==0]
                # Make sure to have balanced data for each attribute (is and isnot) 
                # by only taking maximum num of negative attribute samples equal to num of positive samples
                #not_obj_attr[feat] = random.sample(not_obj_attr[feat],min(len(obj_attr[feat]),len(not_obj_attr[feat])))
            
            self.obj_data.append(full_obj_set)
            self.obj_attr.append(obj_attr)
            self.obj_not_attr.append(not_obj_attr)
            self.pairs.append(rel_pairs)

        perception.resnet.cpu() #remove resnet model from cuda to free up memory
    
    def __len__(self):
        return len(self.obj_data)

    def __getitem__(self,idx):
        # TODO: collate list of dictionaries into dictionary of lists for when idx is a range
        return self.obj_data[idx], self.obj_attr[idx], self.obj_not_attr[idx], self.pairs[idx]



def grouper(n, iterable):
    """
    >>> list(grouper(3, 'ABCDEFG'))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    iterable = iter(iterable)
    return iter(lambda: list(IT.islice(iterable, n)), [])

#my_data = CLEVRGroundingDataset(total_imgs=10, group_size=1, csv_file='scenes_train.json')
#tmp = my_data.__getitem__(1)
