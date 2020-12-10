import os
import json
import logictensornetworks_wrapper as ltnw
import torch
import time
import random
import numpy as np
import itertools as IT
import perception
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_scene_groups = 50 # each subgroup contains 10 scenes
scene_group_size = 10

#ltnw.set_universal_aggreg("pmeaner") # 'hmean', 'mean', 'min', 'pmeaner'
#ltnw.set_existential_aggregator("pmean") # 'max', 'pmean'
#ltnw.set_tnorm("new") # 'min','luk','prod','mean','new'
ltnw.set_layers(12) # TODO: fix loading layer number before trained weights

##################################
### Import data from csv files ###
##################################
start_time = time.time()

with open('../scenes_val.json') as f:
    scenes_json = json.load(f)
    scenes_json = scenes_json['scenes']
    f.close()

def grouper(n, iterable):
    """
    >>> list(grouper(3, 'ABCDEFG'))
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G']]
    """
    iterable = iter(iterable)
    return iter(lambda: list(IT.islice(iterable, n)), [])

random.seed(42) 
split_scenes = list(grouper(scene_group_size, scenes_json))
split_scenes = random.sample(split_scenes, num_scene_groups)

#possible features of an object (excluding 3 pixel-coords)
obj_colors = ['gray','blue','brown','yellow','red','green','purple','cyan']
obj_sizes = ['small','large']
obj_shapes = ['cube','sphere','cylinder']
obj_materials = ['rubber','metal']
## obj_feat : ['gray','blue','brown','yellow','red','green','purple','cyan','small','large','cube','sphere','cylinder','rubber','metal']
obj_feat = obj_colors + obj_sizes + obj_shapes + obj_materials
num_of_features = 512

##############################
### Set up LTN Predicates ###
##############################

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Predicates for Object Features ******')

# Object Features
for feat in obj_feat:
    ltnw.predicate(label=feat.capitalize(), number_of_features_or_vars=num_of_features, device=device)

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Predicates for Spacial Relations ******')

# Spacial Relations
ltnw.predicate(label='Right', number_of_features_or_vars=2*num_of_features, device=device) # Right(?o1,?o2) : o2 is on the right of o1
ltnw.predicate(label='Behind', number_of_features_or_vars=2*num_of_features, device=device) # Behind(?o1,?o2) : o2 is behind o1
ltnw.predicate(label='Front', number_of_features_or_vars=2*num_of_features, device=device) # Front(?o1,?o2) : o2 is in front of o1
ltnw.predicate(label='Left', number_of_features_or_vars=2*num_of_features, device=device) # Left(?o1,?o2) : o2 is on the left of o1

####################
### Load the LTN ###
####################

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Loading saved LTN ******')

#ltnw.initialize_knowledgebase(initial_sat_level_threshold=.99)
ltnw.load_ltn('ltn_library.pt', device=device)

ltnw.set_p_value(2)


##############################
### Set axioms for testing ###
##############################

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Create test axiom dictionary ******')

axioms = {}

for i, feat in enumerate(obj_feat):
    axioms['forall ?is_'+ feat + ' : ' + feat.capitalize() + '(?is_'+ feat + ')'] = []
    axioms['forall ?isnot_'+ feat + ' : ~' + feat.capitalize() + '(?isnot_'+ feat + ')'] = []

# Implicit axioms about object features
## objects can only be one color
for c in obj_colors:
    is_color = ''
    is_not_color = ''
    for not_c in obj_colors:
        if not_c == c: is_color = c.capitalize() + '(?obj)'
        if not_c != c: is_not_color += '~' + not_c.capitalize() + '(?obj) &'
    axioms['forall ?obj: ' + is_color + ' -> ' + is_not_color[:-1]] = []
    axioms['forall ?obj: ' + is_not_color[:-1] + ' -> ' + is_color] = []
## objects can only be one size
for s in obj_sizes:
    is_size = ''
    is_not_size = ''
    for not_s in obj_sizes:
        if not_s == s: is_size = s.capitalize() + '(?obj)'
        if not_s != s: is_not_size += '~' + not_s.capitalize() + '(?obj) &'
    axioms['forall ?obj: ' + is_size + ' -> ' + is_not_size[:-1]] = []
    axioms['forall ?obj: ' + is_not_size[:-1] + ' -> ' + is_size] = []
## objects can only be one shape
for sh in obj_shapes:
    is_shape = ''
    is_not_shape = ''
    for not_sh in obj_shapes:
        if not_sh == sh: is_shape = sh.capitalize() + '(?obj)'
        if not_sh != sh: is_not_shape += '~' + not_sh.capitalize() + '(?obj) &'
    axioms['forall ?obj: ' + is_shape + ' -> ' + is_not_shape[:-1]] = []
    axioms['forall ?obj: ' + is_not_shape[:-1] + ' -> ' + is_shape] = []
## objects can only be one material
for m in obj_materials:
    is_material = ''
    is_not_material = ''
    for not_m in obj_materials:
        if not_m == m: is_material = m.capitalize() + '(?obj)'
        if not_m != m: is_not_material += '~' + not_m.capitalize() + '(?obj) &'
    axioms['forall ?obj: ' + is_material + ' -> ' + is_not_material[:-1]] = []
    axioms['forall ?obj: ' + is_not_material[:-1] + ' -> ' + is_material] = []

axioms['forall ?right_pair : Right(?right_pair)'] = []
axioms['forall ?left_pair : ~Right(?left_pair)'] = []

axioms['forall ?behind_pair : Behind(?behind_pair)'] = []
axioms['forall ?front_pair : ~Behind(?front_pair)'] = []

axioms['forall ?front_pair : Front(?front_pair)'] = []
axioms['forall ?behind_pair : ~Front(?behind_pair)'] = []

axioms['forall ?left_pair : Left(?left_pair)'] = []
axioms['forall ?right_pair : ~Left(?right_pair)'] = []

## Implicit Axioms about spacial relations
axioms['forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Left(?obj, ?obj_2)'] = []
#axioms['forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Right(?obj_2, ?obj)'] = []
axioms['forall ?obj, ?obj_2: ~Left(?obj, ?obj_2) -> Right(?obj, ?obj_2)'] = []
#axioms['forall ?obj, ?obj_2: ~Right(?obj_2, ?obj) -> Right(?obj, ?obj_2)'] = []
#axioms['forall ?obj: ~Right(?obj, ?obj)'] = []

axioms['forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~Right(?obj, ?obj_2)'] = []
#axioms['forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~Left(?obj_2, ?obj)'] = []
axioms['forall ?obj, ?obj_2: ~Right(?obj, ?obj_2) -> Left(?obj, ?obj_2)'] = []
#axioms['forall ?obj, ?obj_2: ~Left(?obj_2, ?obj) -> Left(?obj, ?obj_2)'] = []
#axioms['forall ?obj: ~Behind(?obj, ?obj)'] = []

axioms['forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~Behind(?obj, ?obj_2)'] = []
#axioms['forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~Front(?obj_2, ?obj)'] = []
axioms['forall ?obj, ?obj_2: ~Behind(?obj, ?obj_2) -> Front(?obj, ?obj_2)'] = []
#axioms['forall ?obj, ?obj_2: ~Front(?obj_2, ?obj) -> Front(?obj, ?obj_2)'] = []
#axioms['forall ?obj: ~Front(?obj, ?obj)'] = []

axioms['forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~Front(?obj, ?obj_2)'] = []
#axioms['forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~Behind(?obj_2, ?obj)'] = []
axioms['forall ?obj, ?obj_2: ~Front(?obj, ?obj_2) -> Behind(?obj, ?obj_2)'] = []
#axioms['forall ?obj, ?obj_2: ~Behind(?obj_2, ?obj) -> Behind(?obj, ?obj_2)'] = []
#axioms['forall ?obj: ~Left(?obj, ?obj)'] = []

## Check for negations (these should be =0)
# for i, feat in enumerate(obj_feat):
#     axioms['forall ?is_'+ feat + ' : ~' + feat.capitalize() + '(?is_'+ feat + ')'] = []
#     axioms['forall ?isnot_'+ feat + ' : ' + feat.capitalize() + '(?isnot_'+ feat + ')'] = []

#######################################
### Parse JSON data and Test Axioms ###
#######################################
start_time = time.time()
s_time = time.time()
print('******* Testing on JSON subgroup data ******')
pbar = tqdm.tqdm(total=num_scene_groups)

for scenes_subset in split_scenes:
    full_obj_set = [] #list of all objects from all scenes
    full_obj_feat = [] #list of all features per object
    right_pairs = [] #list of index pairs [o1,o2] where o2 is to the right of o1 
    behind_pairs = [] #list of index pairs [o1,o2] where o2 is behind of o1 
    front_pairs = [] #list of index pairs [o1,o2] where o2 is to the in front of o1 
    left_pairs = [] #list of index pairs [o1,o2] where o2 is to the left of o1 

    for idx, scene in enumerate(scenes_subset):

        # Skip over scenes with unclear annotations (num of objects dont equal num of masks)
        if len(scene['objects']) != len(scene['objects_detection']) : continue

        # Number of objects parsed from data (used for indexing of object relations)
        num_obj = len(full_obj_set) 

        # build set of objects in an image (image 0)
        # scene_obj_set = []
        scene_obj_feat = []
        for idx, o in enumerate(scene['objects']):
            color_vec = [(o['color'] == c)*1 for c in obj_colors]
            size_vec = [(o['size'] == s)*1 for s in obj_sizes]
            shape_vec = [(o['shape'] == sh)*1 for sh in obj_shapes]
            material_vec =[(o['material'] == m)*1 for m in obj_materials]
            #pixel_vec = [o['pixel_coords'][0]/480, o['pixel_coords'][1]/320, o['pixel_coords'][0]/32]
            # object features (used to categories and collect groups of same-featured objects)
            scene_obj_feat.append(color_vec + size_vec + shape_vec + material_vec)
            full_obj_feat.append(color_vec + size_vec + shape_vec + material_vec)
            # set of objects in current scene, and of all objects witnessed
            #scene_obj_set.append(perception.get_vector(scene,idx))
            full_obj_set.append(perception.get_vector(scene,idx, mode= 'val'))
        #print('All objects: ', obj_set)

        # right relationship for each object in an image (image 0)
        for i in range(len(scene_obj_feat)):
            for j in range(len(scene['relationships']['right'][i])):
                r_pair = [num_obj+i, num_obj+scene['relationships']['right'][i][j]]    
                right_pairs.append(r_pair)
        #print('Right pairs: ', right_pairs)

        # behind relationship for each object in an image (image 0)   
        for i in range(len(scene_obj_feat)):
            for j in range(len(scene['relationships']['behind'][i])):
                b_pair = [num_obj+i, num_obj+scene['relationships']['behind'][i][j]]
                behind_pairs.append(b_pair)
        #print('Behind pairs: ', behind_pairs)

        # front relationship for each object in an image (image 0)   
        for i in range(len(scene_obj_feat)):
            for j in range(len(scene['relationships']['front'][i])):
                f_pair = [num_obj+i, num_obj+scene['relationships']['front'][i][j]]
                front_pairs.append(f_pair)
        #print('Front pairs:', front_pairs)

        # left relationship for each object in an image (image 0)
        for i in range(len(scene_obj_feat)):
            for j in range(len(scene['relationships']['left'][i])):
                l_pair = [num_obj+i, num_obj+scene['relationships']['left'][i][j]]
                left_pairs.append(l_pair)
        #print('Left pairs:', left_pairs)

    ### Create Subsets of object attributes
    obj_attr, not_obj_attr = {}, {}
    for i, feat in enumerate(obj_feat):
        obj_attr[feat] = [x for (idx, x) in enumerate(full_obj_set) if full_obj_feat[idx][i]==1]
        not_obj_attr[feat] = [x for (idx, x) in enumerate(full_obj_set) if full_obj_feat[idx][i]==0]
        # Make sure to have balanced data for each attribute (is and ~is) 
        # by only taking maximum num of negative attribute samples equal to num of positive samples
        #not_obj_attr[feat] = random.sample(not_obj_attr[feat],min(len(obj_attr[feat]),len(not_obj_attr[feat])))

    ##################
    ### Set Up LTN ###
    ##################
    # time_diff = time.time()-start_time
    # print('Time to complete : ', time_diff)
    # start_time = time.time() 
    # print('******* Setting up LTN ******')    

    num_of_features = len(full_obj_set[0]) # =512 (output of resnet-32 layer3 for whole image (256) + object (256))

    # Object Constants/Variables
    #for i in range(len(full_obj_set)):
    #    ltnw.constant('object'+str(i),full_obj_set[i])

    # 'verbose' argument is used to bypass the variable redeclare warning message
    ltnw.variable('?obj',torch.stack(full_obj_set), verbose=False)
    ltnw.variable('?obj_2',torch.stack(full_obj_set), verbose=False)
    for i, feat in enumerate(obj_feat):
        ltnw.variable('?is_'+feat, torch.stack(obj_attr[feat]), verbose=False)
        ltnw.variable('?isnot_'+feat, torch.stack(not_obj_attr[feat]), verbose=False)
    ltnw.variable('?right_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in right_pairs]), verbose=False)
    ltnw.variable('?left_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in left_pairs]), verbose=False)
    ltnw.variable('?front_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in front_pairs]), verbose=False)
    ltnw.variable('?behind_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in behind_pairs]), verbose=False)

    ## Test the axioms on the freshly declared variables
    with torch.no_grad():
        for a in axioms.keys():
            axioms[a].append(ltnw.ask(a))

    axioms_mean = {k:sum(axioms[k])/len(axioms[k]) for k in axioms.keys()}
    all_axioms_mean = np.array([axioms_mean[k] for k in axioms_mean.keys()]).sum()/len(axioms_mean)
    pbar.set_description("Current Mean : %f" % (all_axioms_mean))
    pbar.update(1)

axioms_mean = {k:sum(axioms[k])/len(axioms[k]) for k in axioms.keys()}
axioms_min = {k:min(axioms[k]) for k in axioms.keys()}
axioms_max = {k:max(axioms[k]) for k in axioms.keys()}

all_axioms_mean = np.array([axioms_mean[k] for k in axioms_mean.keys()]).sum()/len(axioms_mean)

for k in axioms_mean:
    print(k, ' = ', axioms_mean[k], '(min: ', axioms_min[k], ', max: ', axioms_max[k], ')')

print('Overall mean : ' , all_axioms_mean)


time_diff = time.time()-start_time
print('Time to complete : ', time_diff)