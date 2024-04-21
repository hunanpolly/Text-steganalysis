from __future__ import print_function, division, absolute_import
import sys
import torch
import numpy 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertModel, BertTokenizer
import os
import argparse
import math
import torch.utils.model_zoo as model_zoo
import mmd
import coral
import L2
import SinkhornDistance
from torch.autograd import Variable
from bert_embedding import BertEmbedding

class MyBert(nn.Module):
   
    def __init__(self, args):
        super(MyBert, self).__init__()
        self.args = args

        self.bert = args.model 
        for param in self.bert.parameters():
            param.requires_grad = False

    

    def forward(self, x):
        context = x[0]
        mask = x[2]
        outputs = self.bert(context, attention_mask=mask)
        encoder_outputs, text_cls = outputs[0], outputs[1]

        return encoder_outputs

   


class LS_CNN(nn.Module):
    def __init__(self, inplanes,inplanes2,args, channels1,channels2,reduction1,reduction2, gate_eta,text_field=None):
        super(LS_CNN, self).__init__()


        D = inplanes#emb_dim
        D2 = inplanes2#emb_dim
        Ci = 1
        Co = 100#Co = args.kernel_num
        Ks = [3,4,5]

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv2d(Ci,300,(D2,1))])
        self.dropout = nn.Dropout(args.dropout)

        self.fc0 = nn.Conv2d(channels1, channels1 // reduction1, kernel_size=1,padding=0)
        self.fc1 = nn.Conv2d(channels1, channels1 // reduction1, kernel_size=1,padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels2 // reduction1, channels2, kernel_size=1,padding=0)
        self.sigmoid = nn.Sigmoid()

        self.gate = False
        self.gate_eta = gate_eta
        self.feature_maps = {}
        #self.fc1 = nn.Linear(len(Ks)*Co, C)
    

    def forward(self, x,y):
		#x  (batch，seq_len,hidden_size)
        x = x.unsqueeze(3)#在第四维增加一个维度
        x = x.permute(0,3,1,2)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        y = y.unsqueeze(3)#在第四维增加一个维度
        y = y.permute(0,3,1,2)
        y = [F.relu(conv(y)).squeeze(3) for conv in self.convs1]
        y = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]

        module_inputx = x
        module_inputy = y



        x = torch.cat(x, 1)
        x = self.dropout(x)
        y = torch.cat(y, 1)
        y = self.dropout(y)

        self.feature_maps['x_1'] = x.detach().cpu().numpy()
        self.feature_maps['y_1'] = y.detach().cpu().numpy()




        y = y.view(y.size(0), -1)
        x = x.view(x.size(0), -1)

        mmdL_loss = mmd.mmd(x, y)
        coral_loss = coral.CORAL(x, y)
        prob = (2.718282**(mmdL_loss*10))*coral_loss

          

        module_inputx = torch.stack(module_inputx, 1)
        module_inputy = torch.stack(module_inputy, 1)
        if self.training:
            module_inputx = module_inputx.unsqueeze(3)
            module_inputy = module_inputy.unsqueeze(3)
            module_inputx = self.fc0(module_inputx)
            if prob < self.gate_eta:
                self.gate = True
                module_inputy = self.fc0(module_inputy)
            else:
                self.gate = False
                module_inputy = self.fc1(module_inputy)
        else:
            module_inputx = module_inputx.unsqueeze(3)
            module_inputy = module_inputy.unsqueeze(3)
            if self.gate:
                module_inputx = self.fc0(module_inputx)
                module_inputy = self.fc0(module_inputy)
            else:
                module_inputx = self.fc1(module_inputx)
                module_inputy = self.fc1(module_inputy)


        module_input = torch.cat((module_inputx,module_inputy), 1)

        module_input = self.relu(module_input)
        module_input = self.fc2(module_input)
        module_input = self.sigmoid(module_input)

        indices0=torch.tensor([0])
        indices0=indices0.to('cuda')
        indices1=torch.tensor([1])
        indices1=indices1.to('cuda')
        indices2=torch.tensor([2])
        indices2=indices2.to('cuda')
        indices3=torch.tensor([3])
        indices3=indices3.to('cuda')
        indices4=torch.tensor([4])
        indices4=indices4.to('cuda')
        indices5=torch.tensor([5])
        indices5=indices5.to('cuda')


        module_input1 = torch.index_select(module_input, 1,indices0)
        module_input2 = torch.index_select(module_input, 1,indices1)
        module_input3 = torch.index_select(module_input, 1,indices2)
        module_inputx = torch.cat((module_input1,module_input2,module_input3), 2)

        module_input4 = torch.index_select(module_input, 1,indices3)
        module_input5 = torch.index_select(module_input, 1,indices4)
        module_input6 = torch.index_select(module_input, 1,indices5)
        module_inputy = torch.cat((module_input4,module_input5,module_input6), 2)



        module_inputx = [F.relu(conv(module_inputx)).squeeze(3) for conv in self.convs2]
        module_inputx = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in module_inputx]

        module_inputy = [F.relu(conv(module_inputy)).squeeze(3) for conv in self.convs2]
        module_inputy = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in module_inputy]

        module_inputx = torch.cat(module_inputx, 1)
        module_inputx = self.dropout(module_inputx)
        module_inputy = torch.cat(module_inputy, 1)
        module_inputy = self.dropout(module_inputy)

        self.feature_maps['x_2'] = module_inputx.detach().cpu().numpy()
        self.feature_maps['y_2'] = module_inputy.detach().cpu().numpy()
   

        return  x * module_inputx,y * module_inputy




class MFSAN(nn.Module): 

    def __init__(self, args):
        super(MFSAN, self).__init__()
        self.sharedNet = MyBert(args)
        self.sonnet1 = LS_CNN(768,100,args,3,6,3,6,0.002)
        self.cls_fc_son1 = nn.Linear(300, 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.training = True

    def forward(self, data_src, data_tgt=0, label_src=0, mark = 1):
        mmd_loss = 0
        if self.training == True:

            
            if mark == 1:#第一个源域
                data_src = self.sharedNet(data_src)
                data_tgt = self.sharedNet(data_tgt)




                data_src,data_tgt_son1 = self.sonnet1(data_src,data_tgt)
                data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
                data_src = data_src.view(data_src.size(0), -1)

                mmdL_loss = mmd.mmd(data_src, data_tgt_son1)
                coral_loss = coral.CORAL(data_src, data_tgt_son1)
                mmd_loss += (2.718282**(mmdL_loss*10))*coral_loss


                pred_src = self.cls_fc_son1(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)


                return cls_loss, mmd_loss


        else:#测试阶段对目标域的预测

            data = self.sharedNet(data_src)

            fea_son1,fea_son2 = self.sonnet1(data,data)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            return pred1


