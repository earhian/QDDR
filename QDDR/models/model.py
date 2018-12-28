import torch
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
from models.modelZoo import *
from losses.main import *
from torch.nn.parallel.data_parallel import data_parallel


def get_weaky_label(label, num_classes=340):
    b = len(label)
    zeros = torch.zeros((b, num_classes)).cuda().float()
    for i in range(b):
        zeros[i, label[i]] = 1
    return zeros
class model_ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        # self.moe1 = MoeModel(feature_size=2048, num_classes=512, num_mixture=2)
        # self.moe2 = MoeModel(feature_size=1536, num_classes=512, num_mixture=2)
        # self.fc = MoeModel(feature_size=2048, num_classes=340, num_mixture=2)
        self.fc0 = nn.Linear(2048, 512)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(2048, 512)
        self.fc5 = nn.Linear(1536, 512)
        self.fc6 = nn.Linear(2048, 512)
        self.fc7 = nn.Linear(2048, 512)
        self.fc = nn.Linear(512 * 8 , 340)
        self.fc_sum = nn.Linear(512  , 340)
    def forward(self, x0, x1,x2, x3, x4, x5, x6, x7, countryCode):
        x0 = self.fc0(x0)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
        x4 = self.fc4(x4)
        x5 = self.fc5(x5)
        x6 = self.fc6(x6)
        x7 = self.fc7(x7)
        x0 = F.dropout(x0, p=0.2)
        x1 = F.dropout(x1, p=0.2)
        x2 = F.dropout(x2, p=0.2)
        x3 = F.dropout(x3, p=0.2)
        x4 = F.dropout(x4, p=0.2)
        x5 = F.dropout(x5, p=0.2)
        x6 = F.dropout(x6, p=0.2)
        x7 = F.dropout(x7, p=0.2)
        # countryCode = get_weaky_label(countryCode, num_classes=218)
        x = self.fc(torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], 1))
        x_sum = self.fc_sum(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7)
        return x + x_sum

    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
    def getLoss(self, result, target, loss_function=nn.CrossEntropyLoss()):
        # self.loss = nn.CrossEntropyLoss()(target, result.long())
        self.loss = loss_function(result, target.long())
class model_QDDR(nn.Module):
    def __init__(self, num_classes=340, get_feature=False, inchannels=3,model_name='resnet34'):
        super().__init__()
        if model_name == 'resnet34':
            self.basemodel = tvm.resnet34(True)
            if inchannels == 1:
                self.basemodel.conv1 = nn.Conv2d(inchannels, 64, kernel_size=5, stride=2,padding=2)
            self.basemodel.avgpool = nn.AdaptiveAvgPool2d(1)
            self.basemodel.fc = nn.Sequential(
                                nn.Dropout(0.2),
                                nn.Linear(512, num_classes)
                                              )
        elif model_name == 'xception':
            self.basemodel = xception(True, get_feature=get_feature)
        elif model_name == 'inceptionv4':
            self.basemodel = inceptionv4(num_classes=num_classes, get_feature=get_feature)
        elif model_name == 'shufflenetv2':
            self.basemodel = shufflenetv2()
        elif model_name == 'mobilenet':
            self.basemodel = MobileNet()
        elif model_name == 'dpn68':
            self.basemodel = dpn68(num_classes=num_classes, pretrained=True)
        elif model_name == 'I3d':
            self.basemodel = get_I3d(True)
        elif model_name == 'seresnext50':
            self.basemodel = se_resnext50_32x4d( inchannels=inchannels, get_feature=get_feature, pretrained='imagenet')
        elif model_name == 'seresnext101':
            self.basemodel = se_resnext101_32x4d(num_classes=num_classes, get_feature=get_feature,pretrained='imagenet')
        elif model_name == 'inceptionresnetv2':
            self.basemodel = inceptionresnetv2(num_classes=num_classes)
        else:
            assert False, "{} is error".format(model_name)
    def forward(self, x):
        x = data_parallel(self.basemodel, x)
        return x

    def getLoss(self,  result, target, loss_function=nn.CrossEntropyLoss()):
        # self.loss = nn.CrossEntropyLoss()(target, result.long())
        self.loss = loss_function(result, target.long())

    def mean_teacher_loss(self, outs_student, outs_teacher, recognized, labels, con_weight=1):
        b = len(recognized)
        isrecognized_index = torch.nonzero(recognized).view(-1)
        nonrecognized_index = torch.nonzero(recognized == 0).view(-1)
        self.loss = (len(isrecognized_index)/b) * nn.CrossEntropyLoss()(outs_student[isrecognized_index], labels[isrecognized_index].long())
        len_nonrecognized = len(nonrecognized_index)
        if len_nonrecognized > 0:
            outs_teacher = Variable(outs_teacher.data, requires_grad=False)
            labels_teacher = labels[nonrecognized_index]
            outs_s = outs_student[nonrecognized_index]
            weaky_label = get_weaky_label(labels_teacher)
            target = (torch.softmax(outs_teacher, 1) + weaky_label)/2
            target = torch.clamp(target, 0.0, 1.0)
            self.loss += (len(nonrecognized_index)/b) * F.binary_cross_entropy(torch.softmax(outs_s, 1), target) * con_weight



    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        # raise NotImplementedError


