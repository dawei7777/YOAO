
import time
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision.models as models
from torchvision import datasets 
from torch.utils.data import DataLoader  
from YOAO import YOAO
from DeepFool import deepfool
# 设定设备  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
  
# 加载ResNet-18 模型  
model_ResNet18 = models.resnet18(pretrained=False) 
num_ftrs = model_ResNet18.fc.in_features  
model_ResNet18.fc = torch.nn.Linear(num_ftrs, 10)  # 将输出层的神经元数量改为10  
model_ResNet18 = model_ResNet18.cuda()   
model_ResNet18.load_state_dict(torch.load('RESNET_cifar10_3.pth')) 
model_ResNet18.eval() 

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

def run_attack(attack_type,image, net, num_classes, overshoot, max_iter):
    if attack_type == 1:
        r, loop_i, label_orig, label_pert, pert_image = YOAO(image, net, num_classes, overshoot, max_iter)
    elif attack_type == 2:
        r, loop_i, label_orig, label_pert, pert_image = deepfool(image, net, num_classes, overshoot, max_iter)
    else:
        raise ValueError("Invalid attack type. Please choose 1 for YOAO or 2 for DeepFool.")
    return r, loop_i, label_orig, label_pert, pert_image

max_iter = 1
overshoot = 1
num_classes=10
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
test_data = datasets.CIFAR10(root='./data/', train=False,download=True, 
                             transform= transforms.Compose([  
                                               transforms.ToTensor(),  
                                               transforms.Normalize(mean=mean, std=std)  # 归一化
                                               ])  )


class_labels_en = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

attack_type = int(input("请选择攻击类型，输入数字1、2（1: YOAO, 2: DeepFool）: "))
for m in range(1):
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)  # 创建一个数据加载器 
    count=0
    total=0
    right=0
    p_total=0
    start_time = time.time()
    for images, labels in test_loader:  

        count=count+1
        images, labels = images.to(device), labels.to(device)  
        for i, (image, label) in enumerate(zip(images, labels)):
            r, loop_i, label_orig, label_pert, pert_image = run_attack(attack_type,image, model_ResNet18, num_classes, overshoot,max_iter)
            print("Original label = ", label_orig)
            print("Perturbed label = ", label_pert) 
            if label_orig!=label_pert:
                right=right+1
            # 计算图像 image 的 2-范数
                image_cpu = image.cpu()
                p_adv = np.linalg.norm(r.flatten(), ord=2)  # 计算 2-范数    
                p_total=p_adv+p_total


            break   
        if count>500:
            p_advv=p_total/count
            break
    end_time = time.time()
    execution_time = end_time - start_time
    print("代码运行时间：", execution_time, "秒")       
    print("ACC：", (right/count)*100, "%")   
    print("count：", count-1, "张图片")   
    print("p_advv", p_advv, "")   
