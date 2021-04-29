import os
import argparse
import torch
from trades_without_ce_loss import *
from cifar10_models import *
import torchvision
from torchvision import datasets, transforms
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

prefix = 'resnet18-teacher_wideresnet_adv_0.56_loss_1_power_2e4_single_teachers_attack2_shuffle_300_epochs_ce_1_5_2_new_attack_method'
epochs = 300
lr = 0.1
epsilon = 8/255.0
batch_size = 128
attack_iters = 10

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

student = resnet18()
student = torch.nn.DataParallel(student)
student = student.cuda()
student.train()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
kl_loss = nn.KLDivLoss(size_average=False,reduce=False)
def ce_loss(a,b):
    loss = -a*b
    return loss
teacher = wideresnet()
teacher.load_state_dict(torch.load('./models/success_model/model_cifar_wrn.pt'))
teacher = torch.nn.DataParallel(teacher)
teacher = teacher.cuda()
teacher.eval()

for epoch in range(1,epochs+1):
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(train_batch_data)

        adv_logits = trades_loss_without_celoss6(student,teacher_logits,train_batch_data,train_batch_labels,optimizer,step_size=0.0078,epsilon=epsilon,perturb_steps=10)
        student.train()
        nat_logits = student(train_batch_data)
        kl_Loss1 = ce_loss(F.log_softmax(adv_logits,dim=1),F.softmax(teacher_logits.detach(),dim=1))
        kl_Loss2 = ce_loss(F.log_softmax(nat_logits,dim=1),F.softmax(teacher_logits.detach(),dim=1))
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
        torch.save({'student':student.state_dict(),'optimizer':optimizer.state_dict()},'./models/'+prefix+str(np.sum(test_accs==0)/len(test_accs))+'.pth')
    if epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
