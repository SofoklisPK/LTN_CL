import os
import json
import logictensornetworks_wrapper as ltnw
import torch


##################################
### Import data from csv files ###
##################################

with open('scenes_short.json') as f:
    scenes_json = json.load(f)
    f.close()

with open('questions_short.json') as f:
    questions_json = json.load(f)
    f.close()

#print('first scene:\n',scenes_json[0])
#print('##########\n##########\nfirst question:\n',questions_json[0])


#######################
### Load datapoints ###
#######################

#possible features of an object (excluding 3 pixel-coords)
obj_colors = ['gray','blue','brown','yellow','red','green','purple','cyan']
obj_sizes = ['small','large']
obj_shapes = ['cube','sphere','cylinder']
obj_materials = ['rubber','metal']
## obj_feat : ['gray','blue','brown','yellow','red','green','purple','cyan','small','large','cube','sphere','cylinder','rubber','metal']
obj_feat = obj_colors + obj_sizes + obj_shapes + obj_materials
num_of_features = len(obj_feat) + 3
## num_of features: 18 , The extra 3 are the pixel coordinates (did not include 3d-coords and rotation yet)

#build set of objects in an image (image 0)
obj_set = []
for o in scenes_json[0]['objects']:
    color_vec = [(o['color'] == c)*1 for c in obj_colors]
    size_vec = [(o['size'] == s)*1 for s in obj_sizes]
    shape_vec = [(o['shape'] == sh)*1 for sh in obj_shapes]
    material_vec =[(o['material'] == m)*1 for m in obj_materials]
    obj_set.append(color_vec + size_vec + shape_vec + material_vec + o['pixel_coords'])
#print('All objects: ', obj_set)

# right relationship for each object in an image (image 0)
right_pairs = [] #list of pairs [o1,o2] where o2 is to the right of o1 
for i in range(len(obj_set)):
    for j in range(len(scenes_json[0]['relationships']['right'][i])):
        r_pair = [i, scenes_json[0]['relationships']['right'][i][j]]
        right_pairs.append(r_pair)
#print('Right pairs: ', right_pairs)

# behind relationship for each object in an image (image 0)
behind_pairs = [] #list of pairs [o1,o2] where o2 is behind of o1 
for i in range(len(obj_set)):
    for j in range(len(scenes_json[0]['relationships']['behind'][i])):
        b_pair = [i, scenes_json[0]['relationships']['behind'][i][j]]
        behind_pairs.append(b_pair)
#print('Behind pairs: ', behind_pairs)

# front relationship for each object in an image (image 0)
front_pairs = [] #list of pairs [o1,o2] where o2 is to the in front of o1 
for i in range(len(obj_set)):
    for j in range(len(scenes_json[0]['relationships']['front'][i])):
        f_pair = [i, scenes_json[0]['relationships']['front'][i][j]]
        front_pairs.append(f_pair)
#print('Front pairs:', front_pairs)

# left relationship for each object in an image (image 0)
left_pairs = [] #list of pairs [o1,o2] where o2 is to the left of o1 
for i in range(len(obj_set)):
    for j in range(len(scenes_json[0]['relationships']['left'][i])):
        l_pair = [i, scenes_json[0]['relationships']['left'][i][j]]
        left_pairs.append(l_pair)
#print('Left pairs:', left_pairs)


##################
### Set Up LTN ###
################## 
       
#ltnw.set_universal_aggreg("hmean")
#ltnw.set_existential_aggregator("max")
#ltnw.set_tnorm("luk")
ltnw.set_layers = 4
max_epochs = 15000

# Object Constants/Variables
for i in range(len(obj_set)):
    ltnw.constant('object'+str(i),obj_set[i])
ltnw.variable('?obj',obj_set)
ltnw.variable('?obj_2',obj_set)
#print(ltnw.CONSTANTS)

# Object Features (TODO)
for feat in obj_feat:
    ltnw.predicate(feat.capitalize(), num_of_features)

# Implicit axioms about object features 
# (TODO)

# Spacial Relations
ltnw.predicate('Right',2*num_of_features)
ltnw.predicate('Behind',2*num_of_features)
ltnw.predicate('Front',2*num_of_features)
ltnw.predicate('Left',2*num_of_features)

# Axioms for image's spacial relationships
for p in right_pairs:
    ltnw.axiom('Right(object'+str(p[0])+',object'+str(p[1])+')')   
for p in behind_pairs:
    ltnw.axiom('Behind(object'+str(p[0])+',object'+str(p[1])+')') 
for p in front_pairs:
    ltnw.axiom('Front(object'+str(p[0])+',object'+str(p[1])+')') 
for p in left_pairs:
    ltnw.axiom('Left(object'+str(p[0])+',object'+str(p[1])+')') 

# Implicit Axioms about spacial relations
ltnw.axiom('forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~ Left(?obj, ?obj_2)')
ltnw.axiom('forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~ Right(?obj_2, ?obj)')

ltnw.axiom('forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~ Right(?obj, ?obj_2)')
ltnw.axiom('forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~ Left(?obj_2, ?obj)')

ltnw.axiom('forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~ Behind(?obj, ?obj_2)')
ltnw.axiom('forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~ Front(?obj_2, ?obj)')

ltnw.axiom('forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~ Front(?obj, ?obj_2)')
ltnw.axiom('forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~ Behind(?obj_2, ?obj)')

#####################
### Train the LTN ###
#####################

ltnw.initialize_knowledgebase(initial_sat_level_threshold=.99)
sat_level = ltnw.train(max_epochs=max_epochs,sat_level_epsilon=.1, early_stop_level=0.00001)

####################
### Test the LTN ###
####################

print('Is object0 (large brown cylinder) in front of object3 (large purple sphere)? ', ltnw.ask('Front(object3,object0)'))
print('Is object4 (small gray cube) not to the left of object3 (large purple sphere)? ', ltnw.ask('~Left(object3,object4)'))
print('Is object2 (small green cylinder) to the left of object1 (large gray cube)? ', ltnw.ask('Left(object1,object2)'))





