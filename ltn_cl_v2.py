import os
import json
import logictensornetworks_wrapper as ltnw
import torch
import time
import random

num_scenes = 5
num_of_layers = 4
max_epochs = 1500

##################################
### Import data from csv files ###
##################################

with open('CLEVR_train_scenes.json') as f:
    scenes_json = json.load(f)
    scenes_json = scenes_json['scenes']
    f.close()

#random.seed(42)
scenes_subset = random.sample(scenes_json, num_scenes)

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
num_of_features = len(obj_feat) + 3
## num_of features: 18 , The extra 3 are the pixel coordinates (did not include 3d-coords and rotation yet)

full_obj_set = [] #list of all objects from all scenes
right_pairs = [] #list of index pairs [o1,o2] where o2 is to the right of o1 
behind_pairs = [] #list of index pairs [o1,o2] where o2 is behind of o1 
front_pairs = [] #list of index pairs [o1,o2] where o2 is to the in front of o1 
left_pairs = [] #list of index pairs [o1,o2] where o2 is to the left of o1 

for idx, scene in enumerate(scenes_subset):

    # Number of objects parsed from data (used for indexing of object relations)
    num_obj = len(full_obj_set) 

    # build set of objects in an image (image 0)
    scene_obj_set = []
    for o in scene['objects']:
        color_vec = [(o['color'] == c)*1 for c in obj_colors]
        size_vec = [(o['size'] == s)*1 for s in obj_sizes]
        shape_vec = [(o['shape'] == sh)*1 for sh in obj_shapes]
        material_vec =[(o['material'] == m)*1 for m in obj_materials]
        scene_obj_set.append(color_vec + size_vec + shape_vec + material_vec + o['pixel_coords'])
        full_obj_set.append(color_vec + size_vec + shape_vec + material_vec + o['pixel_coords'])
    #print('All objects: ', obj_set)

    # right relationship for each object in an image (image 0)
    for i in range(len(scene_obj_set)):
        for j in range(len(scene['relationships']['right'][i])):
            r_pair = [num_obj+i, num_obj+scene['relationships']['right'][i][j]]    
            right_pairs.append(r_pair)
    #print('Right pairs: ', right_pairs)

    # behind relationship for each object in an image (image 0)   
    for i in range(len(scene_obj_set)):
        for j in range(len(scene['relationships']['behind'][i])):
            b_pair = [num_obj+i, num_obj+scene['relationships']['behind'][i][j]]
            behind_pairs.append(b_pair)
    #print('Behind pairs: ', behind_pairs)

    # front relationship for each object in an image (image 0)   
    for i in range(len(scene_obj_set)):
        for j in range(len(scene['relationships']['front'][i])):
            f_pair = [num_obj+i, num_obj+scene['relationships']['front'][i][j]]
            front_pairs.append(f_pair)
    #print('Front pairs:', front_pairs)

    # left relationship for each object in an image (image 0)
    for i in range(len(scene_obj_set)):
        for j in range(len(scene['relationships']['left'][i])):
            l_pair = [num_obj+i, num_obj+scene['relationships']['left'][i][j]]
            left_pairs.append(l_pair)
    #print('Left pairs:', left_pairs)

 

##################
### Set Up LTN ###
##################
time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Setting up LTN ******')    

#ltnw.set_universal_aggreg("hmean")
#ltnw.set_existential_aggregator("max")
#ltnw.set_tnorm("luk")
#ltnw.set_layers(4) # logictensornetworks.py line 277 makes this irrelevant to actual layers used!!


# Object Constants/Variables
for i in range(len(full_obj_set)):
    ltnw.constant('object'+str(i),full_obj_set[i])
ltnw.variable('?obj',full_obj_set)
ltnw.variable('?obj_2',full_obj_set)

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Predicate/Axioms for Object Features ******')
# Object Features
for feat in obj_feat:
    ltnw.predicate(label=feat.capitalize(), number_of_features_or_vars=num_of_features, layers=num_of_layers)

# Axioms for object features in an image
for i in range(len(full_obj_set)):
    for j in range(len(obj_feat)):
        if full_obj_set[i][j] == 1:
            ltnw.axiom(obj_feat[j].capitalize() + '(object' + str(i) + ')')
            #print(obj_feat[j].capitalize() + '(object' + str(i) + ')')
        #else:
        #    ltnw.axiom('~'+obj_feat[j].capitalize() + '(object' + str(i) + ')')

# Implicit axioms about object features
## objects can only be one color
for c in obj_colors:
    is_color = ''
    is_not_color = ''
    for not_c in obj_colors:
        if not_c == c: is_color = 'forall ?obj: ' + c.capitalize() + '(?obj) -> '
        if not_c != c: is_not_color += '~' + not_c.capitalize() + '(?obj) &'
    ltnw.axiom(is_color + is_not_color[:-1])
## objects can only be one size
for s in obj_sizes:
    is_size = ''
    is_not_size = ''
    for not_s in obj_sizes:
        if not_s == s: is_size = 'forall ?obj: ' + s.capitalize() + '(?obj) -> '
        if not_s != s: is_not_size += '~' + not_s.capitalize() + '(?obj) &'
    ltnw.axiom(is_size + is_not_size[:-1])
## objects can only be one shape
for sh in obj_shapes:
    is_shape = ''
    is_not_shape = ''
    for not_sh in obj_shapes:
        if not_sh == sh: is_shape = 'forall ?obj: ' + sh.capitalize() + '(?obj) -> '
        if not_sh != s: is_not_shape += '~' + not_sh.capitalize() + '(?obj) &'
    ltnw.axiom(is_shape + is_not_shape[:-1])
## objects can only be one material
for m in obj_materials:
    is_material = ''
    is_not_material = ''
    for not_m in obj_materials:
        if not_m == m: is_material = 'forall ?obj: ' + m.capitalize() + '(?obj) -> '
        if not_m != m: is_not_material += '~' + not_m.capitalize() + '(?obj) &'
    ltnw.axiom(is_material + is_not_material[:-1])

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Predicate/Axioms for Spacial Relations ******')
# Spacial Relations
ltnw.predicate(label='Right', number_of_features_or_vars=2*num_of_features, layers=num_of_layers) # Right(?o1,?o2) : o2 is on the right of o1
ltnw.predicate(label='Behind', number_of_features_or_vars=2*num_of_features, layers=num_of_layers) # Behind(?o1,?o2) : o2 is behind o1
ltnw.predicate(label='Front', number_of_features_or_vars=2*num_of_features, layers=num_of_layers) # Front(?o1,?o2) : o2 is in front of o1
ltnw.predicate(label='Left', number_of_features_or_vars=2*num_of_features, layers=num_of_layers) # Left(?o1,?o2) : o2 is on the left of o1

# Axioms for image's spacial relationships
for p in right_pairs:
    ltnw.axiom('Right(object'+str(p[0])+',object'+str(p[1])+')')
    ltnw.axiom('~Left(object'+str(p[0])+',object'+str(p[1])+')')
for p in behind_pairs:
    ltnw.axiom('Behind(object'+str(p[0])+',object'+str(p[1])+')') 
    ltnw.axiom('~Front(object'+str(p[0])+',object'+str(p[1])+')') 
for p in front_pairs:
    ltnw.axiom('Front(object'+str(p[0])+',object'+str(p[1])+')')
    ltnw.axiom('~Behind(object'+str(p[0])+',object'+str(p[1])+')') 
for p in left_pairs:
    ltnw.axiom('Left(object'+str(p[0])+',object'+str(p[1])+')')
    ltnw.axiom('~Right(object'+str(p[0])+',object'+str(p[1])+')') 

# Implicit Axioms about spacial relations
ltnw.axiom('forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Left(?obj, ?obj_2)')
ltnw.axiom('forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Right(?obj_2, ?obj)')
ltnw.axiom('forall ?obj: ~Right(?obj, ?obj)')

ltnw.axiom('forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~Right(?obj, ?obj_2)')
ltnw.axiom('forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~Left(?obj_2, ?obj)')
ltnw.axiom('forall ?obj: ~Behind(?obj, ?obj)')

ltnw.axiom('forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~Behind(?obj, ?obj_2)')
ltnw.axiom('forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~Front(?obj_2, ?obj)')
ltnw.axiom('forall ?obj: ~Front(?obj, ?obj)')

ltnw.axiom('forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~Front(?obj, ?obj_2)')
ltnw.axiom('forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~Behind(?obj_2, ?obj)')
ltnw.axiom('forall ?obj: ~Left(?obj, ?obj)')

#####################
### Train the LTN ###
#####################
time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Initialising LTN ******')
ltnw.initialize_knowledgebase(initial_sat_level_threshold=.99)

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)
start_time = time.time() 
print('******* Training LTN ******')
sat_level = ltnw.train(max_epochs=max_epochs,sat_level_epsilon=.1, track_values=True)#, early_stop_level=0.00001)

####################
### Test the LTN ###
####################

# ask queries about objects in image_val_00000.png
print('\nIs object0 (large brown cylinder) in front of object3 (large purple sphere)? ', ltnw.ask('Front(object3,object0)'))
print('Is object3 (large purple sphere) not to the left of object2 (small green cylinder)? ', ltnw.ask('~Left(object2,object3)'))
print('Is object2 (small green cylinder) to the left of object1 (large gray cube)? ', ltnw.ask('Left(object1,object2)'))
print('Is object4 (small gray cube) to the right of object0 (large brown cylinder)? ', ltnw.ask('Right(object0, object4)'))
print('Is object2 (small green cylinder) small? ', ltnw.ask('Small(object2)'))
print('Is object1 (large gray cube) a sphere? ', ltnw.ask('Sphere(object1)'))
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



