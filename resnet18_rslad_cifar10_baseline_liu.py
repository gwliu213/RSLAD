import os
import argparse
import torch
from rslad_loss import *
from cifar10_models import *
import torchvision
from torchvision import datasets, transforms
import time
# we fix the random seed to 0, this method can keep the results consistent in the same conputer.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
use_cuda = torch.cuda.is_available()
criterion_CE = nn.CrossEntropyLoss()
prefix = 'resnet18-CIFAR10_RSLAD'
epochs = 300
batch_size = 128
epsilon = 8/255.0 #perturbation
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
# student = torch.nn.DataParallel(student)
# student = student.cuda()

optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
def kl_loss(a,b):
    loss = -a*b + torch.log(b+1e-5)*b
    return loss
def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total
teacher = wideresnet()

teacher.load_state_dict(torch.load('./models/model_cifar_wrn.pt'))

# teacher = torch.nn.DataParallel(teacher)
# teacher = teacher.cuda()
if torch.cuda.device_count() > 1:
    teacher = nn.DataParallel(teacher).to(device)
    student = nn.DataParallel(student).to(device)
else:
    teacher = teacher.to(device)
    student = student.to(device)
test(teacher)
student.train()
teacher.eval()

for epoch in range(1,epochs+1):
    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        with torch.no_grad():
            # teacher_logits = teacher(train_batch_data)
            t_feats, teacher_logits = teacher.extract_feature(train_batch_data, preReLU=True)

        # adv_logits = rslad_inner_loss(student,teacher_logits,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)
        s_adv_logits, train_batch_data_adv, s_adv_feats = rslad_inner_loss(student, teacher_logits,
                                                                                       train_batch_data,
                                                                                       train_batch_labels, optimizer,
                                                                                       step_size=2 / 255.0,
                                                                                       epsilon=epsilon,
                                                                                       perturb_steps=10)
        student.train()
        nat_logits = student(train_batch_data)
        kl_Loss1 = kl_loss(F.log_softmax(s_adv_logits,dim=1),F.softmax(teacher_logits.detach(),dim=1))
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
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
            logits = student(test_ifgsm_data.cuda())
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs = test_accs + predictions.tolist()
        test_accs = np.array(test_accs)
        test_acc = np.sum(test_accs==0)/len(test_accs)
        print('robust acc',np.sum(test_accs==0)/len(test_accs))
        torch.save(student.state_dict(),'./models_res/'+prefix+str(np.sum(test_accs==0)/len(test_accs))+'.pth')
    if epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
