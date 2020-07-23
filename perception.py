import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import json
import random
import pycocotools.mask as mask


# num_scenes = 5

##################################
### Import data from csv files ###
##################################

# with open('scenes_test.json') as f:
#     scenes_json = json.load(f)
#     scenes_json = scenes_json['scenes']
#     f.close()

# random.seed(3)
# scenes_subset = random.sample(scenes_json, num_scenes)

resnet = models.resnet34(pretrained=True)
## Output of resnet is avgpool of layer3
resnet.layer4, resnet.fc = nn.Identity(), nn.Identity()
#modules=list(resnet.children())[:-1]
#resnet=nn.Sequential(*modules)
#for p in resnet.parameters():
#    p.requires_grad = False

# Set model to evaluation mode
resnet.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(scene, idx):
    image_name = 'images/' + scene['split'] + '/' + scene['image_filename']
    #obj = scene['objects'][idx]
    obj_mask = scene['objects_detection'][idx]['mask']
    obj_bbox = mask.toBbox(obj_mask)
    # 1. Load the whole image  and object subpart of image with Pillow library
    img = Image.open(image_name).convert(mode='RGB')
    obj = img.crop((obj_bbox[0], obj_bbox[1], obj_bbox[0]+obj_bbox[2], obj_bbox[1]+obj_bbox[3]))
    # 2. Create a PyTorch Variable with the transformed image
    img_var = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)) # assign it to a variable
    obj_var = Variable(normalize(to_tensor(scaler(obj))).unsqueeze(0)) # assign it to a variable
    img_features_var = resnet(img_var) # get the output from the last hidden layer of the pretrained resnet
    obj_features_var = resnet(obj_var) # get the output from the last hidden layer of the pretrained resnet
    features = torch.cat([img_features_var.data.flatten(),obj_features_var.data.flatten()]) # get the tensor out of the variable
    return features


# feat_vect = get_vector(scenes_json[0], 0)
# print(feat_vect)