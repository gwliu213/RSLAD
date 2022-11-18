# edited by gaowen, save cam visulized images to folder on single image. need to improve to make it work on batch data. 
# add batch data scorecam 
import os
import argparse
import torch
from rslad_loss import *
from cifar10_models import *
import torchvision
from torchvision import datasets, transforms
import numpy as np
import cv2
import skimage.transform
import torch.nn as nn
import torch.nn. functional as F
import torchvision.transforms.functional as trans_resize
from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torchvision.utils import save_image
from torchvision.transforms import Resize
from cam import CAM, GradCAM
from utils.visualize import visualize, reverse_normalize
from pytorch_grad_cam import ScoreCAM, AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from pytorch_grad_cam.utils.image import show_cam_on_image
# we fix the random seed to 0, this method can keep the results consistent in the same conputer.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

prefix = 'resnet18-CIFAR10_RSLAD'
epochs = 300
batch_size = 16#128
epsilon = 8/255.0 #perturbation

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    # parser.add_argument(
    #     '--image-path',
    #     type=str,
    #     default='/home/gaowen/Documents/NetworkCompression/pytorch-grad-cam/examples/both.png',
    #     help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='scorecam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
def reshape_transform(tensor, height=25, width=25):
    # result = tensor[:, 15:, :].reshape(tensor.size(0),
    #                                   height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    # result = result.transpose(2, 3).transpose(1, 2)
    result = tensor
    return result
def tensor2im(input_image, imtype=np.uint8):
    """"
    Parameters:
        input_image (tensor) --  输入的tensor，维度为CHW，注意这里没有batch size的维度
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.5, 0.5, 0.5] # dataLoader中设置的mean参数，需要从dataloader中拷贝过来
    std = [0.5, 0.5, 0.5]  # dataLoader中设置的std参数，需要从dataloader中拷贝过来
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): # 如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): # 反标准化，乘以方差，加上均值
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 #反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    return image_numpy.astype(imtype)
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])
def saveCam(cam,batch_size,targets,input_tensor,train_batch_data,file_name):
    grayscale_cams = cam(input_tensor=input_tensor,
                         targets=targets,
                         eigen_smooth=args.eigen_smooth,
                         aug_smooth=args.aug_smooth)
    # torchvison.utils.save_image(denormalize(train_batch_data))
    # Here grayscale_cam has only one image in the batch

    for i in range(batch_size):
        grayscale_cam = grayscale_cams[i, :]  # 224 224
        # rgb_img = Function.interpolate(train_batch_data[i].unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False)
        rgb_img = train_batch_data[i].squeeze(0)  # 3,224,224
        img = tensor2im(train_batch_data[i])  # 224,224,3
        big_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        # rgb_img = img.transpose(1, 2, 0)
        cam_image = show_cam_on_image(img / 255, grayscale_cam)  # 224,224,3
        # save_image(rgb_img, str(i) + 'image.png')
        big_cam_image = cv2.resize(cam_image, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./sample/'+file_name+str(i)+'.png', big_img)
        cv2.imwrite('./sample/'+'cam'+file_name+str(i)+'.jpg', big_cam_image)


#
if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {
         "scorecam": ScoreCAM,
         "ablationcam": AblationCAM,
         }

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    transform_train = transforms.Compose(
        [transforms.ToTensor(),
        #torchvision.transforms.ToTensor() accept PIL Image or numpy.ndarray,
         ## first convert HWC to CHW, then convert to float, each pixel /255.
         # transforms.Resize(224),
         transforms.Resize(32),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    
    student = resnet18()
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
    target_layer = [teacher.module.block3.layer[4].conv2]
    # target_layer = teacher.module.block3.layer[4].conv2
    if args.method == "ablationcam":
        cam = methods[args.method](model=teacher,
                                   target_layers=target_layer,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=teacher,
                                   target_layers=target_layer,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)

    for epoch in range(1,epochs+1):
        for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
            # _, _, H, W = train_batch_data.shape
            # input_tensor = train_batch_data
            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            targets = None
            saveCam(cam=cam,batch_size=batch_size,targets=None,input_tensor=train_batch_data,train_batch_data=train_batch_data,file_name='cam_natural')

# ---------------------------------------------------------------------------------------------------
# # single image GradCAM
#             wrapped_model =GradCAM(teacher, target_layer)
#             cam, idx = wrapped_model(train_batch_data[0].unsqueeze(0))
#             # visualize only cam
#             imshow(cam.squeeze().cpu().numrgb_imgpy(), alpha=0.5, cmap='jet')
#             # reverse normalization for display
#             # img = reverse_normalize(train_batch_data[0].unsqueeze(0))
#             heatmap = visualize(train_batch_data[0].unsqueeze(0), cam.cpu())
#             # save image
#             # save_image(heatmap, './sample/{}_cam.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
#             new_heatmap=F.interpolate(heatmap, size=(224, 224), mode='bicubic', align_corners=False)
#             save_image(new_heatmap, './sample/{}_gradcam.png'.format(idx))
#             # or visualize on Jupyter
#             hm = (new_heatmap.squeeze().numpy().transpose(1, 2, 0)).astype(np.int32)
#             imshow(hm)
#             new_ori = F.interpolate(train_batch_data[0].unsqueeze(0), size=(224, 224), mode='bicubic', align_corners=False)
#             save_image(new_ori, './sample/image.png')
            # ---------------------------------------------------------------------------------------------------
            student.train()
            train_batch_data = train_batch_data.float().cuda()
            train_batch_labels = train_batch_labels.cuda()
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(train_batch_data)

            adv_logits,train_batch_data_adv = rslad_inner_loss(student,teacher_logits,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)
            saveCam(cam=cam, batch_size=batch_size, targets=None, input_tensor=train_batch_data_adv,
                    train_batch_data=train_batch_data,file_name='cam_adv')
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
            torch.save(student.state_dict(),'./models/'+prefix+str(np.sum(test_accs==0)/len(test_accs))+'.pth')
        if epoch in [215,260,285]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
