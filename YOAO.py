# 改进deepfool
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import csv
def zero_gradients(variable):  
    if variable.grad is not None:  
        variable.grad.zero_()  
def YOAO(image, net, num_classes, overshoot, max_iter,margin_factor=0):
    image = image.cuda()
    net = net.cuda()
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]
    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    loop_i = 0
    
    max_grad_norm_idx = -1  # 初始化最大梯度范数的索引为无效值  
    max_grad_val = float('-inf')  # 用于存储梯度最大的grad值  
    max_grad_norm = -0.00001  # 初始化为负无穷大  
    
    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
#     fs_list = [fs[0,I[k]] for k in range(num_classes)] # 初始化一个列表，存储每个类别的输出
    k_i = label
     
           # 计算每个类别的梯度  
    for k in range(num_classes):  
        zero_gradients(x)  # 清除之前的梯度  
        fs[0, I[k]].backward(retain_graph=True)  # 对当前类别执pyttho行反向传播  
        grad = x.grad.data.cpu().numpy().copy()  # 获取梯度  
        grad_norm = np.linalg.norm(grad.flatten())  # 计算梯度的L2范数  
        # grad_diff[k] = grad_norm  # 存储梯度范数  
                # 找到梯度差异最大的类别  
        if grad_norm > max_grad_norm:   
            max_grad_norm_idx = k  
            max_grad_val = grad  # 存储梯度最大的grad值   
    
    while k_i == label and loop_i < max_iter :
        
        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
#     对正确标签执行反向传播
        grad_orig = x.grad.data.cpu().numpy().copy()    
#     获得当前标签梯度
        w_k = max_grad_val - grad_orig
#     当前类别和正确类别之间的梯度差
        f_k = (fs[0, I[max_grad_norm_idx]]  - fs[0, I[0]]).data.cpu().numpy()
#     当前类别和正确类别之间的置信度差
        pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
#     取置信度差的绝对值并除以梯度差的L2范数
        if pert_k < pert:
            pert = pert_k
            w = w_k
#     当前类别的扰动值是否小于之前存储的最小扰动值。更新
        r_i =  ((pert+1e-7) * w / np.linalg.norm(w+1e-7))
#     计算当前迭代中的扰动，添加（1e-4）以避免除以零。
        r_tot = np.float32(r_tot + r_i)
#     累加总扰动
        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
#     更新图像，通过添加累积的扰动
        x = Variable(pert_image, requires_grad=True)
#     图像转换为变量，设置其需要梯度
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
#     计算更新后图像的预测类别
        loop_i += 1
        r_tot = (1+overshoot)*r_tot
#     增加扰动幅度，将原有的扰动放大 overshoot 比例
    r_tot = (1+margin_factor)*r_tot
    pert_image = image + torch.from_numpy(r_tot).cuda()
    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    k_i = np.argmax(fs.data.cpu().numpy().flatten())
#     计算更新后图像的预测类别
    return r_tot, loop_i, label, k_i, pert_image












