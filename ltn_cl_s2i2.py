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
### Parse JSON data ###
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

# Object Features
for feat in obj_feat:
    ltnw.predicate(feat.capitalize(), num_of_features)

# Axioms for object features in an image
for i in range(len(obj_set)):
    for j in range(len(obj_feat)):
        if obj_set[i][j] == 1:
            ltnw.axiom(obj_feat[j].capitalize() + '(object' + str(i) + ')')
            #print(obj_feat[j].capitalize() + '(object' + str(i) + ')')

# Implicit axioms about object features
## objects can only be one color
for c in obj_colors:
    for not_c in obj_colors:
        if not_c != c:
            ltnw.axiom('forall ?obj: ' + c.capitalize() + '(?obj) -> ~' + not_c.capitalize() + '(?obj)')
## objects can only be one size
for s in obj_sizes:
    for not_s in obj_sizes:
        if not_s != s:
            ltnw.axiom('forall ?obj: ' + s.capitalize() + '(?obj) -> ~' + not_s.capitalize() + '(?obj)')
## objects can only be one shape
for sh in obj_shapes:
    for not_sh in obj_shapes:
        if not_sh != sh:
            ltnw.axiom('forall ?obj: ' + sh.capitalize() + '(?obj) -> ~' + not_sh.capitalize() + '(?obj)')
## objects can only be one material
for m in obj_materials:
    for not_m in obj_materials:
        if not_m != m:
            ltnw.axiom('forall ?obj: ' + m.capitalize() + '(?obj) -> ~' + not_m.capitalize() + '(?obj)')

# Spacial Relations
ltnw.predicate('Right',2*num_of_features) # Right(?o1,?o2) : o2 is on the right of o1
ltnw.predicate('Behind',2*num_of_features) # Behind(?o1,?o2) : o2 is behind o1
ltnw.predicate('Front',2*num_of_features) # Front(?o1,?o2) : o2 is in front of o1
ltnw.predicate('Left',2*num_of_features) # Left(?o1,?o2) : o2 is on the left of o1

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

ltnw.initialize_knowledgebase(initial_sat_level_threshold=.99)
sat_level = ltnw.train(max_epochs=max_epochs,sat_level_epsilon=.1, early_stop_level=0.00001)

####################
### Test the LTN ###
####################

# ask queries about objects in image_val_00000.png
print('\nIs object0 (large brown cylinder) in front of object3 (large purple sphere)? ', ltnw.ask('Front(object3,object0)'))
print('Is object4 (small gray cube) not to the left of object2 (small green cylinder)? ', ltnw.ask('~Left(object3,object2)'))
print('Is object2 (small green cylinder) to the left of object1 (large gray cube)? ', ltnw.ask('Left(object1,object2)'))
print('Is object4 (small gray cube) to the right of object0 (large brown cylinder)? ', ltnw.ask('Right(object0, object4)'))
print('Is object2 (small green cylinder) small? ', ltnw.ask('Small(object2)'))
print('Is object1 (large gray cube) a sphere? ', ltnw.ask('Sphere(object1)'))





