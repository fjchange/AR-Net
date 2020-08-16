import os
import h5py
import numpy as np
from torch.utils.data import Dataset

class SH_Train_Dataset(Dataset):
    def __init__(self,part_num,h5_file,train_txt,norm=0):
        self.h=h5py.File(h5_file,'r')
        self.train_txt=train_txt
        self.load_feat()
        self.shuffle_keys()
        self.part_num=part_num
        # self.sample_prob=0.8
        self.norm=norm

    def load_feat(self):
        self.norm_feats_dict={}
        self.abnorm_feats_dict={}
        self.norm_feats=[]
        self.abnorm_feats=[]
        self.norm_scenes=[]
        self.abnorm_scenes=[]

        for i in range(1,14):
            self.norm_feats_dict[i]=[]
            self.abnorm_feats_dict[i]=[]

        lines=open(self.train_txt,'r').readlines()
        for line in lines:
            line_split=line.strip().split(',')
            scene=int(line_split[0].split('_')[0])
            label=int(line_split[-1])
            key=line_split[0]+'.npy'

            if label==0:
                self.norm_feats_dict[scene].append(self.h[key].value)
                self.norm_feats.append(self.h[key].value)
                self.norm_scenes.append(scene)
            else:
                self.abnorm_feats_dict[scene].append(self.h[key].value)
                self.abnorm_feats.append(self.h[key].value)
                self.abnorm_scenes.append(scene)
    def __len__(self):
        return min(len(self.abnorm_scenes),len(self.norm_scenes))

    def shuffle_keys(self):
        self.norm_iters=np.random.permutation(len(self.norm_scenes))
        self.abnorm_iters=np.random.permutation(len(self.abnorm_scenes))


    def pad(self,feat):
        feat_len=feat.shape[0]
        return np.concatenate([feat,
                               np.zeros([self.part_num-feat_len,feat.shape[-1]],dtype=np.float32)],axis=0)

    def sample_feat(self,feat,norm=False):
        feat_len=feat.shape[0]
        if feat_len<=self.part_num:
            return self.pad(feat),feat_len
        else:
            if norm:
                begin=np.random.randint(0,feat_len-self.part_num)
                feat=feat[begin:begin+self.part_num]
            else:
                chosen=np.sort(np.random.sample(feat_len)[:self.part_num])
                feat=feat[chosen]
            return feat,self.part_num


    def __getitem__(self, i):
        norm_feat,norm_len=self.sample_feat(self.norm_feats[self.norm_iters[i]],norm=True)
        norm_scene=self.norm_scenes[self.norm_iters[i]]
        abnorm_feat,abnorm_len=self.sample_feat(self.abnorm_feats[self.abnorm_iters[i]])
        abnorm_scene=self.abnorm_scenes[self.abnorm_iters[i]]

        return norm_feat,abnorm_feat,norm_scene,abnorm_scene,norm_len,abnorm_len

def load_shanghaitech_test(txt_path,mask_dir,h5_file,norm):
    lines=open(txt_path,'r').readlines()
    feats=[]
    annos = []
    labels=[]
    names=[]
    h=h5py.File(h5_file)
    for line in lines:
        line_split=line.strip().split(',')
        feat=h[line_split[0]+'.npy']
        if norm==2:
            feat=feat/np.linalg.norm(feat,axis=-1,keepdims=True)
        if line_split[1]=='1':
            anno_npy_path=os.path.join(mask_dir,line_split[0]+'.npy')
            anno=np.load(anno_npy_path)
            # if anno.shape[0]%segment_len!=0:
            #     anno=np.sum(anno[:-(anno.shape[0]%segment_len)].reshape([-1,segment_len]),axis=-1,keepdims=False)
            # else:
            #     anno=np.sum(anno.reshape([-1,segment_len]),axis=-1,keepdims=False)
            # anno=anno.clip(0,1)
            labels.append('Abnormal')
        else:
            anno=np.zeros(int(line_split[-1]))
            labels.append('Normal')
        if anno.shape[0]==0:
            print(line)
        feats.append(feat)
        annos.append(anno)
        names.append(line_split[0])
    return feats,labels,annos
