import os
import json
import logictensornetworks_wrapper as ltnw
import torch
import time
import random
import perception
import tqdm
import csv
import itertools as IT
import dataset
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


total_images = 200
scene_group_size = 10
max_epochs = 5000
learning_rate = 1e-5

ltnw.set_universal_aggreg("pmeaner") # 'hmean', 'mean', 'min', 'pmeaner'
ltnw.set_existential_aggregator("pmean") # 'max', 'pmean'
ltnw.set_tnorm("new") # 'min','luk','prod','mean','new'
#ltnw.set_layers(3)
#ltnw.set_p_value(0.5)
p_factor = 2 # p_value = p_factor*(sat_value**2)

perception_mode = 'val' # potentially set up to backprop to perception module

#######################
### Parse JSON data ###
#######################
start_time = time.time()
s_time = time.time()
print('******* Creating CLEVR Grounded dataset ******')

clevr_dataset = dataset.CLEVRGroundingDataset(total_imgs=total_images, group_size=scene_group_size, csv_file='../scenes_train.json')

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)

##################
### Set Up LTN ###
##################
start_time = time.time() 
print('******* Setting up LTN with Axiom Library ******')    

num_of_features = 512 # =512 (output of resnet-32 layer3 for whole image (256) + object (256))
#possible features of an object (excluding 3 pixel-coords)
obj_colors = ['gray','blue','brown','yellow','red','green','purple','cyan']
obj_sizes = ['small','large']
obj_shapes = ['cube','sphere','cylinder']
obj_materials = ['rubber','metal']
## obj_feat : ['gray','blue','brown','yellow','red','green','purple','cyan','small','large','cube','sphere','cylinder','rubber','metal']
obj_feat = obj_colors + obj_sizes + obj_shapes + obj_materials
obj_directions = ['right','left','front','behind']

# define class categories of attributes (MLP structure)
cat_colors = ['Gray', 'Blue', 'Brown', 'Yellow', 'Red', 'Green', 'Purple', 'Cyan']
Category_Color = ltnw.class_category(class_label='Color', number_of_features=num_of_features, names_of_classes=cat_colors,device=device)
cat_sizes = ['Small', 'Large']
Category_Size = ltnw.class_category(class_label='Size', number_of_features=num_of_features, names_of_classes=cat_sizes,device=device)
cat_shapes = ['Cube', 'Sphere', 'Cylinder']
Category_Shape = ltnw.class_category(class_label='Shape', number_of_features=num_of_features, names_of_classes=cat_shapes,device=device)
cat_materials = ['Rubber', 'Metal']
Category_Material = ltnw.class_category(class_label='Material', number_of_features=num_of_features, names_of_classes=cat_materials,device=device)
cat_horizontal = ['Right', 'Left']
Category_Horizontal = ltnw.class_category(class_label='Horizontal', number_of_features=2*num_of_features, names_of_classes=cat_horizontal,device=device)
cat_vertical = ['Front', 'Behind']
Category_Vertical = ltnw.class_category(class_label='Vertical', number_of_features=2*num_of_features, names_of_classes=cat_vertical,device=device)

# Object Variables Placeholders
ltnw.variable('?obj',torch.zeros(1,num_of_features))
ltnw.variable('?obj_2',torch.zeros(1,num_of_features))
for i, feat in enumerate(obj_colors):
    ltnw.mlp_predicate(label=feat.capitalize(), class_category=Category_Color)
    ltnw.variable('?is_'+feat, torch.zeros(1,num_of_features,device=device))
    ltnw.axiom('forall ?is_'+ feat + ' : ' + feat.capitalize() + '(?is_'+ feat + ')')
    ltnw.variable('?isnot_'+feat, torch.zeros(1,num_of_features,device=device))
    ltnw.axiom('forall ?isnot_'+ feat + ' : ~' + feat.capitalize() + '(?isnot_'+ feat + ')')
for i, feat in enumerate(obj_sizes):
    ltnw.mlp_predicate(label=feat.capitalize(), class_category=Category_Size)
    ltnw.variable('?is_'+feat, torch.zeros(1,num_of_features,device=device))
    ltnw.axiom('forall ?is_'+ feat + ' : ' + feat.capitalize() + '(?is_'+ feat + ')')
    ltnw.variable('?isnot_'+feat, torch.zeros(1,num_of_features,device=device))
    ltnw.axiom('forall ?isnot_'+ feat + ' : ~' + feat.capitalize() + '(?isnot_'+ feat + ')')
for i, feat in enumerate(obj_shapes):
    ltnw.mlp_predicate(label=feat.capitalize(), class_category=Category_Shape)
    ltnw.variable('?is_'+feat, torch.zeros(1,num_of_features,device=device))
    ltnw.axiom('forall ?is_'+ feat + ' : ' + feat.capitalize() + '(?is_'+ feat + ')')
    ltnw.variable('?isnot_'+feat, torch.zeros(1,num_of_features,device=device))
    ltnw.axiom('forall ?isnot_'+ feat + ' : ~' + feat.capitalize() + '(?isnot_'+ feat + ')')   
for i, feat in enumerate(obj_materials):
    ltnw.mlp_predicate(label=feat.capitalize(), class_category=Category_Material)
    ltnw.variable('?is_'+feat, torch.zeros(1,num_of_features,device=device))
    ltnw.axiom('forall ?is_'+ feat + ' : ' + feat.capitalize() + '(?is_'+ feat + ')')
    ltnw.variable('?isnot_'+feat, torch.zeros(1,num_of_features,device=device))
    ltnw.axiom('forall ?isnot_'+ feat + ' : ~' + feat.capitalize() + '(?isnot_'+ feat + ')')   



# Implicit axioms about object features
## objects can only be one color
# for c in obj_colors:
#     is_color = ''
#     is_not_color = ''
#     for not_c in obj_colors:
#         if not_c == c: is_color = c.capitalize() + '(?obj)'
#         if not_c != c: is_not_color += '~' + not_c.capitalize() + '(?obj) &'
#     ltnw.axiom('forall ?obj: ' + is_color + ' -> ' + is_not_color[:-1])
#     #ltnw.axiom('forall ?obj: ' + is_not_color[:-1] + ' -> ' + is_color)
# ## objects can only be one size
# for s in obj_sizes:
#     is_size = ''
#     is_not_size = ''
#     for not_s in obj_sizes:
#         if not_s == s: is_size = s.capitalize() + '(?obj)'
#         if not_s != s: is_not_size += '~' + not_s.capitalize() + '(?obj) &'
#     ltnw.axiom('forall ?obj: ' + is_size + ' -> ' + is_not_size[:-1])
#     #ltnw.axiom('forall ?obj: ' + is_not_size[:-1] + ' -> ' + is_size)
# ## objects can only be one shape
# for sh in obj_shapes:
#     is_shape = ''
#     is_not_shape = ''
#     for not_sh in obj_shapes:
#         if not_sh == sh: is_shape = sh.capitalize() + '(?obj)'
#         if not_sh != sh: is_not_shape += '~' + not_sh.capitalize() + '(?obj) &'
#     ltnw.axiom('forall ?obj: ' + is_shape + ' -> ' + is_not_shape[:-1])
#     #ltnw.axiom('forall ?obj: ' + is_not_shape[:-1] + ' -> ' + is_shape)
# ## objects can only be one material
# for m in obj_materials:
#     is_material = ''
#     is_not_material = ''
#     for not_m in obj_materials:
#         if not_m == m: is_material = m.capitalize() + '(?obj)'
#         if not_m != m: is_not_material += '~' + not_m.capitalize() + '(?obj) &'
#     ltnw.axiom('forall ?obj: ' + is_material + ' -> ' + is_not_material[:-1])
#     #ltnw.axiom('forall ?obj: ' + is_not_material[:-1] + ' -> ' + is_material)

# Spacial Relations
ltnw.variable('?right_pair', torch.zeros(1,2*num_of_features,device=device))
ltnw.variable('?left_pair', torch.zeros(1,2*num_of_features,device=device))
ltnw.variable('?front_pair', torch.zeros(1,2*num_of_features,device=device))
ltnw.variable('?behind_pair', torch.zeros(1,2*num_of_features,device=device))

ltnw.mlp_predicate(label='Right', class_category= Category_Horizontal)
ltnw.mlp_predicate(label='Left', class_category= Category_Horizontal)
ltnw.mlp_predicate(label='Front', class_category= Category_Vertical)
ltnw.mlp_predicate(label='Behind', class_category= Category_Vertical)

ltnw.axiom('forall ?right_pair : Right(?right_pair)')
ltnw.axiom('forall ?left_pair : ~Right(?left_pair)')

ltnw.axiom('forall ?behind_pair : Behind(?behind_pair)')
ltnw.axiom('forall ?front_pair : ~Behind(?front_pair)')

ltnw.axiom('forall ?front_pair : Front(?front_pair)')
ltnw.axiom('forall ?behind_pair : ~Front(?behind_pair)')

ltnw.axiom('forall ?left_pair : Left(?left_pair)')
ltnw.axiom('forall ?right_pair : ~Left(?right_pair)')

# # Implicit Axioms about spacial relations
# ltnw.axiom('forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Left(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj, ?obj_2: Right(?obj, ?obj_2) -> ~Right(?obj_2, ?obj)')
# ltnw.axiom('forall ?obj, ?obj_2: ~Left(?obj, ?obj_2) -> Right(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj, ?obj_2: ~Right(?obj_2, ?obj) -> Right(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj: ~Right(?obj, ?obj)')

# ltnw.axiom('forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~Right(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj, ?obj_2: Left(?obj, ?obj_2) -> ~Left(?obj_2, ?obj)')
# ltnw.axiom('forall ?obj, ?obj_2: ~Right(?obj, ?obj_2) -> Left(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj, ?obj_2: ~Left(?obj_2, ?obj) -> Left(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj: ~Behind(?obj, ?obj)')

# ltnw.axiom('forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~Behind(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj, ?obj_2: Front(?obj, ?obj_2) -> ~Front(?obj_2, ?obj)')
# ltnw.axiom('forall ?obj, ?obj_2: ~Behind(?obj, ?obj_2) -> Front(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj, ?obj_2: ~Front(?obj_2, ?obj) -> Front(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj: ~Front(?obj, ?obj)')

# ltnw.axiom('forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~Front(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj, ?obj_2: Behind(?obj, ?obj_2) -> ~Behind(?obj_2, ?obj)')
# ltnw.axiom('forall ?obj, ?obj_2: ~Front(?obj, ?obj_2) -> Behind(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj, ?obj_2: ~Behind(?obj_2, ?obj) -> Behind(?obj, ?obj_2)')
# ltnw.axiom('forall ?obj: ~Left(?obj, ?obj)')

time_diff = time.time()-start_time
print('Time to complete : ', time_diff)

#####################
### Train the LTN ###
#####################
start_time = time.time() 
print('******* Training LTN ******')

pbar = tqdm.tqdm(total=max_epochs)

f = open('../axioms_values.csv', 'w')
dictw = csv.DictWriter(f, ltnw.AXIOMS.keys())
dictw.writeheader()


## Training Loop
for ep in range(max_epochs):

    ## Iterate through batches of images
    for b in range(len(clevr_dataset)):
        full_obj_set, obj_attr, not_obj_attr, pairs = clevr_dataset.__getitem__(b)

        ltnw.variable('?obj',torch.stack(full_obj_set).to(device), verbose=False)
        ltnw.variable('?obj_2',torch.stack(full_obj_set).to(device), verbose=False)
        for i, feat in enumerate(obj_feat):
            if len(obj_attr[feat]) > 0: 
                ltnw.variable('?is_'+feat, torch.stack(obj_attr[feat]).to(device), verbose=False)
            if len(not_obj_attr[feat]) > 0: 
                ltnw.variable('?isnot_'+feat, torch.stack(not_obj_attr[feat]).to(device), verbose=False)  
        ltnw.variable('?right_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in pairs['right']]).to(device), verbose=False)
        ltnw.variable('?left_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in pairs['left']]).to(device), verbose=False)
        ltnw.variable('?front_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in pairs['front']]).to(device), verbose=False)
        ltnw.variable('?behind_pair', torch.stack([torch.cat([full_obj_set[p[0]],full_obj_set[p[1]]]) for p in pairs['behind']]).to(device), verbose=False)

        if ep+b == 0: # Initialise LTN at very beginning of training
            print('******* Initialising LTN ******')
            sat_level = ltnw.initialize_knowledgebase(initial_sat_level_threshold=.5, device=device, learn_rate=learning_rate, perception_mode=perception_mode)
            print("Initial Satisfiability %f" % (sat_level))
            print("Initial p-Value %f" % (p_factor*(sat_level.item()**2)))
        ltnw.set_p_value(p_factor*(sat_level.item()**2))
        sat_level = ltnw.train(max_epochs=1,sat_level_epsilon=.01, track_values=False, device=device, show_progress=False)#, early_stop_level=0.00001)

    dictw.writerow({key:value.detach().cpu().numpy()[0] for (key, value) in ltnw.AXIOMS.items()})
    if sat_level > 0.997: break
    pbar.set_description("Current Satisfiability %f" % (sat_level))
    pbar.update(1)
print("Final p-Value %f" % (p_factor*(sat_level.item()**2)))

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



