import os
import json
import logictensornetworks_wrapper as ltnw
import torch
import time
import random
import perception

num_scenes = 5
num_of_layers = 1
max_epochs = 1000
learning_rate = 0.01

ltnw.set_universal_aggreg("pmeaner") # 'hmean', 'mean', 'min', 'pmeaner'
ltnw.set_existential_aggregator("pmean") # 'max', 'pmean'
ltnw.set_tnorm("new") # 'min','luk','prod','mean','new'
#ltnw.set_layers(4) # logictensornetworks.py line 277 makes this irrelevant to actual layers used!!

##################################
### Import data from csv files ###
##################################

with open('scenes_train.json') as f:
    scenes_json = json.load(f)
    scenes_json = scenes_json['scenes']
    f.close()

random.seed(7)
scenes_subset = random.sample(scenes_json, num_scenes)
#scenes_subset = scenes_json[0:6] # for testing purposes

#with open('questions_short.json') as f:
#    questions_json = json.load(f)
#    f.close()
#print('first scene:\n',scenes_json[0])
#print('##########\n##########\nfirst question:\n',questions_json[0])

#######################
### Parse JSON data ###
#######################
start_time = time.time()
s_time = time.time()
print('******* Parsing JSON data ******')

#possible features of an object (excluding 3 pixel-coords)
obj_colors = ['gray','blue','brown','yellow','red','green','purple','cyan']
obj_sizes = ['small','large']
obj_shapes = ['cube','sphere','cylinder']
obj_materials = ['rubber','metal']
## obj_feat : ['gray','blue','brown','yellow','red','green','purple','cyan','small','large','cube','sphere','cylinder','rubber','metal']
obj_feat = obj_colors + obj_sizes + obj_shapes + obj_materials
#num_of_features = len(obj_feat) + 3
## num_of features: 18 , The extra 3 are the pixel coordinates (did not include 3d-coords and rotation yet)

full_obj_set = [] #list of all objects from all scenes
full_obj_feat = [] #list of all features per object
right_pairs = [] #list of index pairs [o1,o2] where o2 is to the right of o1 
behind_pairs = [] #list of index pairs [o1,o2] where o2 is behind of o1 
front_pairs = [] #list of index pairs [o1,o2] where o2 is to the in front of o1 
left_pairs = [] #list of index pairs [o1,o2] where o2 is to the left of o1 

for idx, scene in enumerate(scenes_subset):

    # Number of objects parsed from data (used for indexing of object relations)
    num_obj = len(full_obj_set) 

    # build set of objects in an image (image 0)
    #scene_obj_set = []
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
        full_obj_set.append(perception.get_vector(scene,idx))
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
    # Make sure to have balanced data for each attribute (is and isnot) 
    # by only taking maximum num of negative attribute samples equal to num of positive samples
    #not_obj_attr[feat] = random.sample(not_obj_attr[feat],min(len(obj_attr[feat]),len(not_obj_attr[feat])))

##################
### Set Up LTN ###
##################
time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Setting up LTN ******')    


num_of_features = len(full_obj_set[0]) # =512 (output of resnet-32 layer3 for whole image (256) + object (256))

# Object Constants/Variables
#for i in range(len(full_obj_set)):
#    ltnw.constant('object'+str(i),full_obj_set[i])
ltnw.variable('?obj',torch.stack(full_obj_set).numpy())
ltnw.variable('?obj_2',torch.stack(full_obj_set).numpy())
for i, feat in enumerate(obj_feat):
    ltnw.variable('?is_'+feat, torch.stack(obj_attr[feat]).numpy())
    ltnw.variable('?isnot_'+feat, torch.stack(not_obj_attr[feat]).numpy())
ltnw.variable('?right_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in right_pairs]).numpy())
ltnw.variable('?left_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in left_pairs]).numpy())
ltnw.variable('?front_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in front_pairs]).numpy())
ltnw.variable('?behind_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in behind_pairs]).numpy())

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Predicate/Axioms for Object Features ******')

# Object Features
for feat in obj_feat:
    ltnw.predicate(label=feat.capitalize(), number_of_features_or_vars=num_of_features, layers=num_of_layers)

for i, feat in enumerate(obj_feat):
    ltnw.axiom('forall ?is_'+ feat + ' : ' + feat.capitalize() + '(?is_'+ feat + ')')
    ltnw.axiom('forall ?isnot_'+ feat + ' : ~' + feat.capitalize() + '(?isnot_'+ feat + ')')

# Implicit axioms about object features
## objects can only be one color
for c in obj_colors:
    is_color = ''
    is_not_color = ''
    for not_c in obj_colors:
        if not_c == c: is_color = c.capitalize() + '(?obj)'
        if not_c != c: is_not_color += '~' + not_c.capitalize() + '(?obj) &'
    ltnw.axiom('forall ?obj: ' + is_color + ' -> ' + is_not_color[:-1])
    ltnw.axiom('forall ?obj: ' + is_not_color[:-1] + ' -> ' + is_color)
## objects can only be one size
for s in obj_sizes:
    is_size = ''
    is_not_size = ''
    for not_s in obj_sizes:
        if not_s == s: is_size = s.capitalize() + '(?obj)'
        if not_s != s: is_not_size += '~' + not_s.capitalize() + '(?obj) &'
    ltnw.axiom('forall ?obj: ' + is_size + ' -> ' + is_not_size[:-1])
    ltnw.axiom('forall ?obj: ' + is_not_size[:-1] + ' -> ' + is_size)
## objects can only be one shape
for sh in obj_shapes:
    is_shape = ''
    is_not_shape = ''
    for not_sh in obj_shapes:
        if not_sh == sh: is_shape = sh.capitalize() + '(?obj)'
        if not_sh != sh: is_not_shape += '~' + not_sh.capitalize() + '(?obj) &'
    ltnw.axiom('forall ?obj: ' + is_shape + ' -> ' + is_not_shape[:-1])
    ltnw.axiom('forall ?obj: ' + is_not_shape[:-1] + ' -> ' + is_shape)
## objects can only be one material
for m in obj_materials:
    is_material = ''
    is_not_material = ''
    for not_m in obj_materials:
        if not_m == m: is_material = m.capitalize() + '(?obj)'
        if not_m != m: is_not_material += '~' + not_m.capitalize() + '(?obj) &'
    ltnw.axiom('forall ?obj: ' + is_material + ' -> ' + is_not_material[:-1])
    ltnw.axiom('forall ?obj: ' + is_not_material[:-1] + ' -> ' + is_material)
    
time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Predicate/Axioms for Spacial Relations ******')
# Spacial Relations
ltnw.predicate(label='Right', number_of_features_or_vars=2*num_of_features, layers=num_of_layers) # Right(?o1,?o2) : o2 is on the right of o1
ltnw.predicate(label='Behind', number_of_features_or_vars=2*num_of_features, layers=num_of_layers) # Behind(?o1,?o2) : o2 is behind o1
ltnw.predicate(label='Front', number_of_features_or_vars=2*num_of_features, layers=num_of_layers) # Front(?o1,?o2) : o2 is in front of o1
ltnw.predicate(label='Left', number_of_features_or_vars=2*num_of_features, layers=num_of_layers) # Left(?o1,?o2) : o2 is on the left of o1


ltnw.axiom('forall ?right_pair : Right(?right_pair)')
ltnw.axiom('forall ?left_pair : ~Right(?left_pair)')

ltnw.axiom('forall ?behind_pair : Behind(?behind_pair)')
ltnw.axiom('forall ?front_pair : ~Behind(?front_pair)')

ltnw.axiom('forall ?front_pair : Front(?front_pair)')
ltnw.axiom('forall ?behind_pair : ~Front(?behind_pair)')

ltnw.axiom('forall ?left_pair : Left(?left_pair)')
ltnw.axiom('forall ?right_pair : ~Left(?right_pair)')

# # Implicit Axioms about spacial relations
ltnw.axiom('forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Left(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Right(?obj_2, ?obj)')
ltnw.axiom('forall ?obj, ?obj_2: ~Left(?obj, ?obj_2) -> Right(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj, ?obj_2: ~Right(?obj_2, ?obj) -> Right(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj: ~Right(?obj, ?obj)')

ltnw.axiom('forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~Right(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~Left(?obj_2, ?obj)')
ltnw.axiom('forall ?obj, ?obj_2: ~Right(?obj, ?obj_2) -> Left(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj, ?obj_2: ~Left(?obj_2, ?obj) -> Left(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj: ~Behind(?obj, ?obj)')

ltnw.axiom('forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~Behind(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~Front(?obj_2, ?obj)')
ltnw.axiom('forall ?obj, ?obj_2: ~Behind(?obj, ?obj_2) -> Front(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj, ?obj_2: ~Front(?obj_2, ?obj) -> Front(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj: ~Front(?obj, ?obj)')

ltnw.axiom('forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~Front(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~Behind(?obj_2, ?obj)')
ltnw.axiom('forall ?obj, ?obj_2: ~Front(?obj, ?obj_2) -> Behind(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj, ?obj_2: ~Behind(?obj_2, ?obj) -> Behind(?obj, ?obj_2)')
#ltnw.axiom('forall ?obj: ~Left(?obj, ?obj)')

#####################
### Train the LTN ###
#####################
time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Initialising LTN ******')
ltnw.initialize_knowledgebase(initial_sat_level_threshold=.5, learn_rate=learning_rate)

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Training LTN ******')
sat_level = ltnw.train(max_epochs=max_epochs,sat_level_epsilon=.001, track_values=True)#, early_stop_level=0.00001)

####################
### Test the LTN ###
####################

# ask queries about objects in image_val_00000.png
# print('\nIs object0 (large brown cylinder) in front of object3 (large purple sphere)? ', ltnw.ask('Front(object3,object0)'))
# print('Is object3 (large purple sphere) not to the left of object2 (small green cylinder)? ', ltnw.ask('~Left(object2,object3)'))
# print('Is object2 (small green cylinder) to the left of object1 (large gray cube)? ', ltnw.ask('Left(object1,object2)'))
# print('Is object4 (small gray cube) to the right of object0 (large brown cylinder)? ', ltnw.ask('Right(object0, object4)'))
# print('Is object2 (small green cylinder) small? ', ltnw.ask('Small(object2)'))
# print('Is object1 (large gray cube) a sphere? ', ltnw.ask('Sphere(object1)'))
#print('Is there an object to the right of object1 (large gray cube)?', ltnw.ask('exists ?obj: Right(object1,?obj)'))

####################
### Save the LTN ###
####################
time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Saving LTN ******')

ltnw.save_ltn()

time_diff = time.time()-s_time
print('Full time to complete : ', time_diff)
start_time = time.time() 



