import os
import argparse
import torch
from rslad_loss import *
from cifar10_models import *
import torchvision
from torchvision import datasets, transforms
import time
import numpy as np
import skimage.transform
import torch.nn as nn
import torch.nn. functional as F
from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torchvision.utils import save_image
from cam import CAM, GradCAM
from rise import RISE
from utils.visualize import visualize, reverse_normalize
# we fix the random seed to 0, in the same computer, this method can make the results same as before.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

prefix = 'mobilenet_v2-CIFAR10_RSLAD'
epochs = 300
batch_size = 128
epsilon = 8/255.0 # perturbation
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

student = mobilenet_v2()
param_count_student=count_param(student)
print('student parameters = ', param_count_student)
student = torch.nn.DataParallel(student)
student = student.cuda()
student.train()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss
teacher = wideresnet()
teacher.load_state_dict(torch.load('./models/model_cifar_wrn.pt'))
param_count_teacher=count_param(teacher)
print('teacher parameters = ', param_count_teacher)
teacher = torch.nn.DataParallel(teacher)
teacher = teacher.cuda()
teacher.eval()
# the target layer you want to visualize
# target_layer = model.layer4[1].conv2
for epoch in range(1,epochs+1):
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
        _, _, H, W = train_batch_data.shape
        wrapped_model = GradCAM(student, input_size=(H, W))
        with torch.no_grad():
            saliency = wrapped_model(train_batch_data[0])
        saliency = saliency[0]
        # reverse normalization for display
        img = reverse_normalize(train_batch_data.to('cpu'))[0]
        img=img.unsqueeze(0)
        saliency = saliency.view(1, 1, H, W)
        heatmap = visualize(img, saliency)
        # or visualize 
        hm = (heatmap.squeeze().numpy().transpose(1, 2, 0)).astype(np.int32)
        imshow(hm)
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(train_batch_data)

        adv_logits = rslad_inner_loss(student,teacher_logits,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)
        student.train()
        nat_logits = student(train_batch_data)
        kl_Loss1 = kl_loss(F.log_softmax(adv_logits,dim=1),F.softmax(teacher_logits.detach(),dim=1))
        kl_Loss2 = kl_loss(F.log_softmax(nat_logits,dim=1),F.softmax(teacher_logits.detach(),dim=1))
        kl_Loss1 = torch.mean(kl_Loss1)
        kl_Loss2 = torch.mean(kl_Loss2)
        loss = 5/6.0*kl_Loss1 + 1/6.0*kl_Loss2
        loss.backward()
        optimizer.step()
        if step%100 == 0:
            print('loss',loss.item())
    if (epoch%20 == 0 and epoch <215) or (epoch%1 == 0 and epoch >= 215):
        test_accs = []
        student.eval()
        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
            logits = student(test_ifgsm_data)
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs = test_accs + predictions.tolist()
        test_accs = np.array(test_accs)
        test_acc = np.sum(test_accs==0)/len(test_accs)
        print('robust acc',np.sum(test_accs==0)/len(test_accs))
        torch.save(student.state_dict(),'./models_mo/'+prefix+str(np.sum(test_accs==0)/len(test_accs))+'.pth')
    if epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
