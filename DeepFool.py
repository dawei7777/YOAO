# 原版deepfool
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import csv
import os
def zero_gradients(variable):  
    if variable.grad is not None:  
        variable.grad.zero_()  
def deepfool(image, net, num_classes, overshoot, max_iter):
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
    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    
    k_i = label
    while k_i == label and loop_i < max_iter:
        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
#     对正确标签执行反向传播
        grad_orig = x.grad.data.cpu().numpy().copy()
#     获取输入x的梯度转移到CPU，转换为NumPy数组，并复制，获得正确标签梯度
        for k in range(1, num_classes):
            zero_gradients(x)
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()
#     获得当前标签梯度
            w_k = cur_grad - grad_orig
#     当前类别和正确类别之间的梯度差
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
#     当前类别和正确类别之间的置信度差
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())
#     取置信度差的绝对值并除以梯度差的L2范数
            if pert_k < pert:
                pert = pert_k
                w = w_k
#     当前类别的扰动值是否小于之前存储的最小扰动值。更新
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
#     计算当前迭代中的扰动方向，添加（1e-4）以避免除以零。
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
    return r_tot, loop_i, label, k_i, pert_image









