import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.nn. functional as F
import cv2
from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torchvision.utils import save_image

from cam import CAM, GradCAM
from utils.visualize import visualize, reverse_normalize
from utils.imagenet_labels import label2idx, idx2label

# from ImageNet
image = Image.open('./sample/tigercat.jpg')
# image = Image.open('./sample/dogsled.jpg')
imshow(image)
# preprocessing. mean and std from ImageNet
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
# convert image to tensor
tensor = preprocess(image)

# reshape 4D tensor (N, C, H, W)
tensor = tensor.unsqueeze(0)
model = models.resnet18(pretrained=True)
model.eval()
print(model)
# the target layer you want to visualize
target_layer = model.layer4[1].conv2

# wrapper for class activation mapping. Choose one of the following.
# wrapped_model = CAM(model, target_layer)
wrapped_model =GradCAM(model, target_layer)
cam, idx = wrapped_model(tensor)
print(idx2label[idx])
# visualize only cam
imshow(cam.squeeze().numpy(), alpha=0.5, cmap='jet')
# reverse normalization for display
img = reverse_normalize(tensor)
heatmap = visualize(img, cam)
# save image
# save_image(heatmap, './sample/{}_cam.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
save_image(heatmap, './sample/{}_gradcam.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
# or visualize on Jupyter
hm = (heatmap.squeeze().numpy().transpose(1, 2, 0)).astype(np.int32)
imshow(hm)
cv2.waitKey(0)
