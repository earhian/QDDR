import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

from timeit import default_timer as timer
from dataSet.reader import *
from dataSet.transform import *
from models.model import *
import torch
import torch.nn as nn
import time
from utils.file import *
from utils.metric import *

"""
step 1 56 padding to 64
step 2 96 padding to 112
"""
size = 96
d, WIDTH, HEIGHT = 3, 112, 112
def train_collate(batch):

    batch_size = len(batch)
    images_v2 = []
    images_v3 = []
    labels = []
    countryCodes = []
    for b in range(batch_size):
        if batch[b] is None:
            continue
        else:
            images_v2.append(batch[b][0])
            images_v3.append(batch[b][1])
            labels.append(batch[b][2])
            countryCodes.append(batch[b][3])
    images_v2 = torch.stack(images_v2, 0)
    images_v3 = torch.stack(images_v3, 0)
    labels = torch.from_numpy(np.array(labels)).long()
    countryCodes = torch.from_numpy(np.array(countryCodes)).long()
    return images_v2, images_v3, labels, countryCodes

def transform_train(drawing, label, recognized, countryCode):
    image = drawing_to_image_with_color_v2(drawing, H=size,W=size)
    image = image.transpose((2, 0, 1))
    image = random_cropping3d(image, (d, WIDTH, HEIGHT), p=0.5)
    image = random_flip(image, p=0.5)
    image = random_erase(image, p=0.5)
    image = image.copy()
    image_v2 = (torch.from_numpy(image).div(255)).float()

    image = drawing_to_image_with_color_v3(drawing, H=size, W=size)
    image = image.transpose((2, 0, 1))
    image = random_cropping3d(image, (d, WIDTH, HEIGHT), p=0.5)
    image = random_flip(image, p=0.5)
    image = random_erase(image, p=0.5)
    image = image.copy()
    image_v3 = (torch.from_numpy(image).div(255)).float()
    return image_v2, image_v3, label, countryCode

def transform_valid(drawing, label, recognized, countryCode):
    image = drawing_to_image_with_color_v2(drawing, H=size,W=size)
    image = image.transpose((2, 0, 1))
    image = random_cropping3d(image, (d, WIDTH, HEIGHT), p=0)
    image = image.copy()
    image_v2 = (torch.from_numpy(image).div(255)).float()

    image = drawing_to_image_with_color_v3(drawing, H=size, W=size)
    image = image.transpose((2, 0, 1))
    image = random_cropping3d(image, (d, WIDTH, HEIGHT), p=0)
    image = image.copy()
    image_v3 = (torch.from_numpy(image).div(255)).float()
    return image_v2, image_v3, label, countryCode


model_name='xception'
model1 = model_QDDR(inchannels=3, model_name=model_name, get_feature=True).cuda()
model1.eval()

model_name='inceptionv4'
model2 = model_QDDR(inchannels=3, model_name=model_name, get_feature=True).cuda()
model2.eval()

model_name='seresnext50'
model3 = model_QDDR(inchannels=3, model_name=model_name, get_feature=True).cuda()
model3.eval()

model_name='seresnext101'
model4 = model_QDDR(inchannels=3, model_name=model_name, get_feature=True).cuda()
model4.eval()
for param in model1.parameters():
    param.detach_()
for param in model2.parameters():
    param.detach_()
for param in model3.parameters():
    param.detach_()
for param in model4.parameters():
    param.detach_()

model_name='xception'
model5 = model_QDDR(inchannels=3, model_name=model_name, get_feature=True).cuda()
model5.eval()

model_name='inceptionv4'
model6 = model_QDDR(inchannels=3, model_name=model_name, get_feature=True).cuda()
model6.eval()

model_name='seresnext50'
model7 = model_QDDR(inchannels=3, model_name=model_name, get_feature=True).cuda()
model7.eval()

model_name='seresnext101'
model8 = model_QDDR(inchannels=3, model_name=model_name, get_feature=True).cuda()
model8.eval()

for param in model5.parameters():
    param.detach_()
for param in model6.parameters():
    param.detach_()
for param in model7.parameters():
    param.detach_()
for param in model8.parameters():
    param.detach_()


model1.load_pretrain('./input/xception_model.pth', skip=[])
model2.load_pretrain('./input/inceptionv4_model.pth', skip=[])
model3.load_pretrain('./input/seresnext50_model.pth', skip=[])
model4.load_pretrain('./input/seresnext101_model.pth', skip=[])
model5.load_pretrain('./input/xception_v3_model.pth', skip=[])
model6.load_pretrain('./input/inceptionv4_v3_model.pth', skip=[])
model7.load_pretrain('./input/seresnext50_v3_model.pth', skip=[])
model8.load_pretrain('./input/seresnext101_v3_model.pth', skip=[])

model = model_ensemble().cuda()

i = 0
iter_smooth = 100
iter_valid = 500
epoch = 0
iter_save = 500
iter_visual = 50
batch_size = 256 * 4
checkPoint_start = 5000

lr = 0.001
resultDir = './result/ensemble_v2_v3'
ImageDir = resultDir + '/image'
checkPoint = os.path.join(resultDir, 'checkpoint')
os.makedirs(checkPoint, exist_ok=True)
os.makedirs(ImageDir, exist_ok=True)
log = Logger()
log.open(os.path.join(resultDir, 'log_train.txt'), mode= 'a')
log.write(' start_time :{} \n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
log.write(' batch_size :{} \n'.format(batch_size))

# model.load_pretrain(os.path.join(checkPoint.replace('_v3', ''), '%08d_model.pth' % (2000)), skip=[])

dst_valid = QDDRDataset(None,num_preclass=('all', 1000), mode='valid',transform=transform_valid)
dataloader_valid = DataLoader(dst_valid,drop_last=False, batch_size=batch_size, num_workers=8, collate_fn=train_collate)
optimizer = torch.optim.SGD(model.parameters(), lr=lr,  momentum=0.9,weight_decay=0.0002)
all_losses = 0
train_loss = 0.0
train_loss = 0.0
valid_loss = 0.0
top1, top3, map3 = 0, 0, 0
top1_train, top3_train, map3_train = 0, 0, 0
top1_batch, top3_batch, map3_batch = 0, 0, 0

batch_loss = 0.0
train_loss_sum = 0
train_top1_sum = 0
train_index = 0

sum = 0
num_train_sum = 0
epoch = 0
skips = []
if not checkPoint_start == 0:
    log.write('start from l_rate ={}, get_learning_rate(optimizer){} \n'.format(lr,checkPoint_start))
    model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)),skip=skips)
    ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
    optimizer.load_state_dict(ckp['optimizer'])
    adjust_learning_rate(optimizer, lr)
    i = checkPoint_start
    epoch = ckp['epoch']
    train_index = ckp['train_index']
log.write(
        ' rate     iter   epoch  | valid   top@1    top@3    map@3  | '
        'train    top@1    top@3    map@3 |'
        ' batch    top@1    top@3    map@3 |  time          \n')
log.write(
        '---------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
start = timer()
def eval(model, dataLoader_valid):
    with torch.no_grad():
        model.eval()
        model.mode = 'valid'
        valid_loss, index_valid= 0, 0
        top1, top3, map3 = 0,0,0
        for valid_data in dataLoader_valid:
            images_v2, images_v3, labels, countryCodes = valid_data
            images_v2 = images_v2.cuda()
            images_v3 = images_v3.cuda()
            labels = labels.cuda()
            countryCodes = countryCodes.cuda()
            feature1 = model1(images_v2)
            feature2 = model2(images_v2)
            feature3 = model3(images_v2)
            feature4 = model4(images_v2)
            feature5 = model5(images_v3)
            feature6 = model6(images_v3)
            feature7 = model7(images_v3)
            feature8 = model8(images_v3)
            results = data_parallel(model, (feature1, feature2, feature3, feature4,
                                            feature5, feature6, feature7, feature8,countryCodes))
            b = len(images_v2)
            top1_, top3_ = accuracy(results, labels, topk=(1,3))
            map3_ = mapk(labels, results, k=3)
            model.getLoss(results,labels)
            valid_loss += model.loss.data.cpu().numpy() * b
            top1 += top1_ * b
            top3 += top3_ * b
            map3 += map3_ * b
            index_valid += b


        valid_loss /= index_valid
        top1 /= index_valid
        top3 /= index_valid
        map3 /= index_valid


        return valid_loss, top1, top3, map3
def get_lr(epoch, lr_min=0.001,lr_max=0.01, cycle_epochs=50):
    epoch = (epoch % cycle_epochs)/cycle_epochs
    if epoch >= 0.5:
        lr = lr_max - (lr_max - lr_min) * (2 * epoch - 1)
    else:
        lr = lr_max - (lr_max - lr_min) * (1 - 2 * epoch)
    return lr
num_shuffle_csv = 997
while i < 10000000:
    dst_train = QDDRDataset(train_index % num_shuffle_csv,num_preclass=('all', 1000), mode='train', transform=transform_train)
    dataloader_train = DataLoader(dst_train, batch_size=batch_size, num_workers=8, collate_fn=train_collate)
    train_index += 1
    for data in dataloader_train:
        epoch = train_index/num_shuffle_csv
        if i % iter_valid == 0:
            valid_loss, top1, top3, map3 = \
                eval(model, dataloader_valid)

            print('\r', end='', flush=True)

            log.write(
                '%0.5f %5.2f k %5.2f  |'
                ' %0.3f    %0.3f    %0.3f    %0.4f | %0.3f    %0.3f    %0.3f | %0.3f     %0.3f    %0.3f | %s \n' % ( \
                    lr, i / 1000, epoch,
                    valid_loss, top1, top3,map3,
                    train_loss, top1_train, 0.0,
                    batch_loss, top1_batch, 0.0,
                    time_to_str((timer() - start) / 60)))
            time.sleep(0.01)

        if i % iter_save == 0 and not i == checkPoint_start:
            torch.save(model.state_dict(), resultDir + '/checkpoint/%08d_model.pth' % (i))
            torch.save({
                'optimizer': optimizer.state_dict(),
                'iter': i,
                'train_index': train_index,
                'epoch': epoch,
            }, resultDir + '/checkpoint/%08d_optimizer.pth' % (i))

        model.train()

        model.mode = 'train'
        images_v2, images_v3, labels, countryCodes = data
        images_v2 = images_v2.cuda()
        images_v3 = images_v3.cuda()
        labels = labels.cuda()
        countryCodes = countryCodes.cuda()
        feature1 = model1(images_v2)
        feature2 = model2(images_v2)
        feature3 = model3(images_v2)
        feature4 = model4(images_v2)
        feature5 = model5(images_v3)
        feature6 = model6(images_v3)
        feature7 = model7(images_v3)
        feature8 = model8(images_v3)
        results = data_parallel(model, (feature1, feature2, feature3, feature4,
                                        feature5, feature6, feature7, feature8, countryCodes))
        model.getLoss(results, labels)
        batch_loss = model.loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        top1_batch = accuracy(results, labels, topk=(1,))[0]
        batch_loss =  batch_loss.data.cpu().numpy()
        sum += 1
        train_loss_sum += batch_loss
        train_top1_sum += top1_batch
        if i%iter_smooth == 0:
            train_loss = train_loss_sum/sum
            top1_train = train_top1_sum/sum
            train_loss_sum = 0
            train_top1_sum = 0
            sum = 0



        print('\r%0.5f %5.2f k %5.2f  | %0.3f    %0.3f    %0.3f    %0.4f | %0.3f    %0.3f    %0.3f | %0.3f     %0.3f    %0.3f | %s  %d %d' % ( \
                lr, i / 1000, epoch,
                valid_loss, top1, top3,map3,
                train_loss, top1_train, 0.0,
                batch_loss, top1_batch, 0.0,
                time_to_str((timer() - start) / 60), checkPoint_start, i)
            , end='', flush=True)
        i += 1






    pass


