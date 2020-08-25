import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import json
import random
import pycocotools.mask as mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
resnet.layer4, resnet.fc = nn.Identity(), nn.Identity() ## Output of resnet is avgpool of layer3
resnet.to(device)

mrcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#mrcnn.roi_heads.box_head, mrcnn.roi_heads.mask_head = nn.Identity(), nn.Identity()
mrcnn.to(device)
tmpin, tmpout = None,None
def hook_fn(self, input, output):
    global tmpin, tmpout
    tmpin = input
    tmpout = output
    print('tmp foo')
mrcnn.roi_heads.box_head.register_forward_hook(hook_fn)

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def detect_objects(filepath):
    """ Dectect objects in an image and return set of RoI bounding boxes
    Args:
        image_filename
    Returns:
        bboxes: list of detected objects bounding boxes in scene (output from RoI Align)
    """

    img = Image.open(filepath).convert(mode='RGB')
    t_img = to_tensor(img).to(device)

    mrcnn.eval()
    ouput = mrcnn([t_img])
    print('foo')



#x = detect_objects('CLEVR_val_000000.png')


def get_vector(scene, idx, mode='val'):
    """ Render ResNet-32 representation of a single object
        Args: 
            scene : the scene annotation TODO: change to filepath & bbox coordinates
            idx : idx of object in scene
            mode : 'train' or 'val' allows for gradient chart and backprop of ResNet model
        Returns:
            features : vector with feature maps of object and whole scene concatenated
    """

    image_name = 'images/' + scene['split'] + '/' + scene['image_filename']
    #obj = scene['objects'][idx]
    obj_mask = scene['objects_detection'][idx]['mask']
    obj_bbox = mask.toBbox(obj_mask)
    # 1. Load the whole image  and object subpart of image with Pillow library
    img = Image.open(image_name).convert(mode='RGB')
    obj = img.crop((obj_bbox[0], obj_bbox[1], obj_bbox[0]+obj_bbox[2], obj_bbox[1]+obj_bbox[3]))
    # 2. Create a PyTorch Variable with the transformed image
    img_var = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0)).to(device) # assign it to a variable
    obj_var = Variable(normalize(to_tensor(scaler(obj))).unsqueeze(0)).to(device) # assign it to a variable
    if mode == 'train' :
        resnet.train()
        img_features_var = resnet(img_var) # get the output from the last hidden layer of the pretrained resnet
        obj_features_var = resnet(obj_var) # get the output from the last hidden layer of the pretrained resnet
    else: 
        resnet.eval()
        with torch.no_grad():
            img_features_var = resnet(img_var) # get the output from the last hidden layer of the pretrained resnet
            obj_features_var = resnet(obj_var) # get the output from the last hidden layer of the pretrained resnet
    features = torch.cat((torch.flatten(img_features_var),torch.flatten(obj_features_var))) # get the tensor out of the variable
    if mode == 'train': return features
    else: return features.cpu().detach()


# feat_vect = get_vector(scenes_json[0], 0)
# print(feat_vect)

#myPerception = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#print(myPerception)