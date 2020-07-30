import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import pairwise_distances

#releated paper:(AAAI2020) Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View.
#https://aaai.org/ojs/index.php/AAAI/article/view/5747

#the numpy version for mad (Be able to compute quickly)
#in_arr:[node_num * hidden_dim], the node feature matrix;
#mask_arr: [node_num * node_num], the mask matrix of the target raltion;
#target_idx = [1,2,3...n], the nodes idx for which we calculate the mad_gap value;
def mad_value(in_arr, mask_arr, distance_metric='cosine', digt_num=4, target_idx =None):
    dist_arr = pairwise_distances(in_arr, in_arr, metric=distance_metric)
    
    mask_dist = np.multiply(dist_arr,mask_arr)

    divide_arr = (mask_dist != 0).sum(1) + 1e-8

    node_dist = mask_dist.sum(1) / divide_arr

    if target_idx.any()==None:
        mad = np.mean(node_dist)
    else:
        node_dist = np.multiply(node_dist,target_idx)
        mad = node_dist.sum()/((node_dist!=0).sum()+1e-8)

    mad = round(mad, digt_num)

    return mad

#the tensor version for mad_gap (Be able to transfer gradients)
#intensor: [node_num * hidden_dim], the node feature matrix;
#neb_mask,rmt_mask:[node_num * node_num], the mask matrices of the neighbor and remote raltion;
#target_idx = [1,2,3...n], the nodes idx for which we calculate the mad_gap value;
def mad_gap_regularizer(intensor,neb_mask,rmt_mask,target_idx):
    node_num,feat_num = intensor.size()

    input1 = intensor.expand(node_num,node_num,feat_num)
    input2 = input1.transpose(0,1)

    input1 = input1.contiguous().view(-1,feat_num)
    input2 = input2.contiguous().view(-1,feat_num)

    simi_tensor = F.cosine_similarity(input1,input2, dim=1, eps=1e-8).view(node_num,node_num)
    dist_tensor = 1 - simi_tensor

    neb_dist = torch.mul(dist_tensor,neb_mask)
    rmt_dist = torch.mul(dist_tensor,rmt_mask)
    
    divide_neb = (neb_dist!=0).sum(1).type(torch.FloatTensor).cuda() + 1e-8
    divide_rmt = (rmt_dist!=0).sum(1).type(torch.FloatTensor).cuda() + 1e-8

    neb_mean_list = neb_dist.sum(1) / divide_neb
    rmt_mean_list = rmt_dist.sum(1) / divide_rmt

    neb_mad = torch.mean(neb_mean_list[target_idx])
    rmt_mad = torch.mean(rmt_mean_list[target_idx])

    mad_gap = rmt_mad - neb_mad

    return mad_gap
