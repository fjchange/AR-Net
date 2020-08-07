import sys
sys.path.append('..')
from Datasets.load_feature import DataLoaderX
import numpy as np
import argparse
import random
import os
from torch import nn
from utils.eval_utils import *
from utils.utils import set_seeds,show_params
from SHTech.AR_Net import *

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,default=2048)
    parser.add_argument('--norm',type=int,default=0)
    parser.add_argument('--segment_len',type=int,default=16)
    parser.add_argument('--batch_size',type=int,default=30)
    parser.add_argument('--part_num',type=int,default=260)
    parser.add_argument('--topk',type=int,default=4)

    parser.add_argument('--iters',type=int,default=50000)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--lr',type=float,default=1e-4)

    parser.add_argument('--dropout_rate',type=float,default=0.7)
    parser.add_argument('--seed',type=int,default=2020)

    parser.add_argument('--model_path_pre',type=str,default='/data0/jiachang/Weakly_Supervised_VAD/models/SHT_')
    parser.add_argument('--training_txt',type=str,default='/data0/jiachang/Weakly_Supervised_VAD/Datasets/SH_Train_new.txt')
    parser.add_argument('--testing_txt',type=str,default='/data0/jiachang/Weakly_Supervised_VAD/Datasets/SH_Test_NEW.txt')
    parser.add_argument('--h5_file',type=str,default='/data0/jiachang/I3D_Joint_shanghai_16_224.h5')
    parser.add_argument('--test_mask_dir',type=str,default='/data0/jiachang/Weakly_Supervised_VAD/Datasets/test_frame_mask/')
    args=parser.parse_args()

    return args
    
def mean(in_list):
    return sum(in_list)/len(in_list)
    
def train_AR_Net(args):
    def worker_init(worked_id):
        np.random.seed(args.seed)
        random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataset=SHT_Dataset.SH_Train_Dataset(args.part_num,args.h5_file,args.training_txt,args.norm)
    dataloader=DataLoaderX(dataset,batch_size=args.batch_size,shuffle=False,num_workers=10,drop_last=True,worker_init_fn=worker_init)
    test_feats,test_labels,test_annos=SHT_Dataset.load_shanghaitech_test(args.testing_txt,args.test_mask_dir,args.h5_file,args.segment_len)
    model=AR_Net(args.size).cuda().train()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=0)
    bce_loss=torch.nn.BCELoss().cuda()
    best_AUC=0
    best_iter=0
    AUCs=[]
    for epoch in range(args.epochs):
        errs=[]
        cens=[]
        labels=torch.cat([torch.zeros(args.batch_size),torch.ones(args.batch_size)],dim=0).cuda().long()
        for norm_feats,abnorm_feats,norm_scenes,abnorm_scenes,norm_lens,abnorm_lens in dataloader:
            feats=torch.cat([norm_feats,abnorm_feats],dim=0).cuda().float().view([args.batch_size*2,args.part_num,args.size])
            seq_lens=torch.cat([norm_lens,abnorm_lens],dim=0).cuda().long()
            scores=model(feats)
            loss,dmil_err,cen_err=DMIL_Cen_Loss(scores,seq_lens,4,bce_loss,args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            errs.append(dmil_err.detach().cpu().numpy())
            cens.append(cen_err.detach().cpu().numpy())
        print('epoch {}\tloss \t {:.4f}\tcen\t{:.4f}'.format(epoch,
                                                              mean(errs),
                                                              mean(cens),
                                                              ))
        dataloader.dataset.shuffle_keys()

        if epoch%10==0:
            score_dict = {}
            label_dict = {}
            total_scores = []
            total_labels = []

            score_dict['Normal'] = []
            score_dict['Abnormal'] = []
            label_dict['Normal'] = []
            label_dict['Abnormal'] = []

            with torch.no_grad():
                model = model.eval()
                for feat, test_label, test_anno in zip(test_feats, test_labels, test_annos):
                    feat=torch.from_numpy(np.array(feat)).cuda().float()
                    score = model(feat).squeeze().detach().cpu().numpy()
                    score=np.repeat(np.expand_dims(score,axis=-1),args.segment_len,axis=-1).reshape([-1])

                    _score=np.zeros_like(test_anno,dtype=np.float32)
                    _score[:score.shape[0]]=score
                    total_labels += test_anno.tolist()
                    total_scores += _score.tolist()
                    label_dict[test_label] = test_anno.tolist()
                    score_dict[test_label] = _score.tolist()
                auc = eval(total_scores, total_labels, label_dict, score_dict)
                AUCs.append(auc)
                if len(AUCs) >= 5:
                    mean_auc = sum(AUCs[-5:]) / 5.
                    if mean_auc > best_AUC:
                        best_iter = epoch
                        best_AUC = mean_auc
                    print('best_AUC {} at epoch {}, now {}'.format(best_AUC, best_iter, mean_auc))
                print('===================')
                model = model.train()

def eval(total_scores,total_labels,label_dict,score_dict):
    total_scores = np.array(total_scores)
    total_labels = np.array(total_labels)

    print('===================')
    normal_far, _ = eval_each_part(label_dict, score_dict)
    auc = cal_auc(total_scores, total_labels)
    pr_auc = cal_pr_auc(total_scores, total_labels)
    acc = cal_accuracy(total_scores, total_labels)
    pre = cal_precision(total_scores, total_labels)
    rec = cal_recall(total_scores, total_labels)
    far = cal_false_alarm(total_scores, total_labels)
    spe = cal_specific(total_scores, total_labels)
    rmse = cal_rmse(total_scores, total_labels)
    gap = cal_score_gap(total_scores, total_labels)
    gm = cal_geometric_mean(total_scores, total_labels)
    mcc = cal_MCC(total_scores, total_labels)
    sen = cal_sensitivity(total_scores, total_labels)
    f = cal_f_measure(total_scores, total_labels)
    pauc = cal_pAUC(total_scores, total_labels)
    fnr = cal_false_neg(total_scores, total_labels)
    print('AUC\t {}\tPR_AUC\t{}\tpAUC\t{}'.format(auc, pr_auc, pauc))
    print('FAR\t{}\tFNR\t{}\tGM\t{}'.format(far, fnr, gm))
    print('Precision\t{}\tRecall\t{}'.format(pre, rec))
    print('Acc\t{}\tMCC\t{}'.format(acc, mcc))
    print('Sen\t{}\tSpe\t{}'.format(sen, spe))
    print('Gap\t{}\tRMSE\t{}'.format(gap, rmse))
    print('F\t{}'.format(f))
    return auc

if __name__=='__main__':
    args=parse_args()
    set_seeds(args.seed)
    show_params(args)
    train_AR_Net(args)
    show_params(args)
