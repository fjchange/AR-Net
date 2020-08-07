import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append('..')

class AR_Net(nn.Module):
    def __init__(self,input_dim):
        super(AR_Net, self).__init__()

        self.FC1=nn.Linear(input_dim,input_dim)
        self.Dropout=nn.Dropout(0.7)
        self.FC2=nn.Linear(input_dim,1)
        torch.nn.init.xavier_uniform_(self.FC1.weight)
        torch.nn.init.xavier_uniform_(self.FC2.weight)

        # weights_normal_init(self.Regressor)

    def forward(self,x):
        return torch.sigmoid(self.FC2(self.Dropout(F.relu(self.FC1(x)))))


def DMIL_Cen_Loss(preds,lens,topk,bce_loss,args):
    dmil_err=0
    cen_err=0
    for i in range(args.batch_size):
        pred=preds[i][:lens[i]].squeeze()
        cen_err+=torch.var(pred)
        topk_num=torch.ceil(lens[i].float()/topk).long().item()
        topk_pred=torch.topk(pred,topk_num)[0]
        dmil_err+=bce_loss(topk_pred,torch.zeros_like(topk_pred).cuda().float())
    for i in range(args.batch_size,args.batch_size*2):
        pred=preds[i][:lens[i]].squeeze()
        topk_num=torch.ceil(lens[i].float()/topk).long().item()
        topk_pred=torch.topk(pred,topk_num)[0]
        dmil_err+=bce_loss(topk_pred,torch.ones_like(topk_pred).cuda().float())

    loss=dmil_err/(args.batch_size*2)+20*cen_err/(args.batch_size)
    return loss,dmil_err,cen_err
