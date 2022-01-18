import numpy as np
import argparse
import os
import time
import datetime
import torch
import torch.nn
import scipy.misc as sm
from utils import decode
from utils.core_utils import *
from torch.utils.data import DataLoader
from utils.metric import runningScore
from dataset.dataset import CityScapesDataset
from model.UNet import UNET

def test(data_loader, Net, loss_fn, log_file, vis_path, testing_score, task):
    Net.eval()
    testing_loss = 0
    time_testing = None
    for i ,(data, target) in enumerate(TestingLoader):
        data, target = data.cuda(), target.cuda()
        timeStart = time.time()
        pred = Net(data)
        timeEnd = time.time()
        if time_testing is None:
            time_testing = np.array([timeEnd-timeStart])
        else:
            time_testing = np.append(time_testing, timeEnd-timeStart)
        testing_loss = loss_fn(pred,target).item()
        pred = pred.data.max(1)[1]
        testing_score.update(target.data.cpu().numpy(), pred.data.cpu().numpy())
        score, class_iou = testing_score.get_scores()

        if vis_path is not None:
            pred = pred.data.cpu().numpy()
            img_pred = decode.decode_segmap(pred,task)
            data = sm.imresize(np.squeeze(data.data.cpu().numpy()).transpose(1,2,0),target.size()[1:3])/255
            img_pred = data*0.5+img_pred*0.5
            # img_pred = np.array(img_pred,dtype=np.uint8)
            sm.imsave(os.path.join(vis_path,'%04d.png'%(i)),img_pred)
    time_testing = np.array(time_testing)
    time_mean = time_testing.mean()
    time_std = time_testing.std()
    time_testing[time_testing>(time_mean+time_std)] = 0
    index_0 = np.where(time_testing==0)
    time_testing = np.delete(time_testing,index_0)
    print_with_write(log_file ,'[---: ---/---] %10s loss: %f OverallAcc: %f MeanAcc %f mIoU %f fps: %f'
            %(('validation'), testing_loss, score['OverallAcc'], score['MeanAcc'], score['mIoU'],
                (time_testing.shape)/time_testing.sum()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img-size',    nargs='+', type=int, default=[256, 512], help='resize to imgsize')
    parser.add_argument('-o', '--output-path', type=str,            default='log',      help='output directory(including log and savemodel)')
    parser.add_argument('-m', '--model-ckpt',  type=str,            default='.//log//savemodel//model_20.pth',       help='the file name of checkpoint you want to test')
    parser.add_argument('-t', '--task',        type=str,            default='cat',      help='the training task: category')
    parser.add_argument('-v', '--vis' ,        type=bool,           default=False,      help='decode the prediction to segmap or not')
    opt = parser.parse_args()

    img_size    = opt.img_size
    model_ckpt  = opt.model_ckpt
    output_path = opt.output_path
    task        = opt.task
    vis         = opt.vis

    assert task in ['cat'],'wrong value of task'
    if task=='cat':
        num_classes = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testing_score = runningScore(num_classes)
    if vis:
        if not os.path.exists(os.path.join(output_path,'predictions')):
            os.makedirs(os.path.join(output_path,'predictions'))
        vis_path = os.path.join(output_path,'predictions')
    else:
        vis_path = None

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_file = open(os.path.join(output_path,'log_test.txt'),'w')
    print_with_write(log_file,str(datetime.datetime.now()))
    print_with_write(log_file,str(opt))

    TestingDataset = CityScapesDataset('data','testing',img_size,task=task)
    TestingLoader  = DataLoader(TestingDataset,batch_size=1,shuffle=False)

    Net = UNET(testing=True)
    Net = Net.cuda()
    
    Net, _, _ = load_model(Net, None, model_ckpt, log_file)
    Net = Net.to(device)
    print_with_write(log_file,'Done!')

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

    test(TestingLoader, Net, loss_fn, log_file, vis_path, testing_score, task)
    print_with_write(log_file,str(datetime.datetime.now()))
