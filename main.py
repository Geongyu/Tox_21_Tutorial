import torch.utils.data
import argparse
import os
import time
import shutil
import pickle
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch import nn as nn
from dataloader import tox_21
from utils import Logger, AverageMeter, save_checkpoint, select_arch, select_optimizer, Performance

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default="/home/deepbio/Desktop/ADMET_code/Data/tox21.csv")
parser.add_argument('--work-dir', default="/home/deepbio/Desktop/ADMET_code")
parser.add_argument('--exp',default="[NR-AR]_ADAM_LR00001_Except_NaN", type=str)

parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--initial-lr', default=0.0001, type=float)
parser.add_argument('--loss-function', default='bce', type=str) 
parser.add_argument('--optim-function', default='adam', type=str)
parser.add_argument('--num-workers', default=1, type=int)
parser.add_argument('--arch', default='BESTox', type=str)
parser.add_argument('--file-name', default='result_train_s_test_s', type=str)
args = parser.parse_args()

def main():
    work_dir = os.path.join(args.work_dir, args.exp)
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    train_filename = args.data_root
    trainset = tox_21(train_filename, mode="train", ratio=0.3, target="NR-AR")
    
    test_set = tox_21(train_filename, mode="test", ratio=0.3, target="NR-AR")

    train_set, valid_set = torch.utils.data.random_split(trainset, [int(len(trainset)*0.8),
                                                     len(trainset)-int(len(trainset)*0.8)])

    args.batch_size = int(args.batch_size)

    train_loader = data.DataLoader(train_set, batch_size=int(args.batch_size),
                                    num_workers=args.num_workers, shuffle=True)

    valid_loader = data.DataLoader(valid_set, batch_size=int(args.batch_size),
                                    num_workers=args.num_workers, shuffle=True)
    
    test_loader = data.DataLoader(test_set, batch_size=int(args.batch_size),
                                    num_workers=args.num_workers, shuffle=True)

    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))
    tst_logger = Logger(os.path.join(work_dir, 'test.log'))

    net = select_arch(args.arch, in_shape=(200, 56))

    # loss
    # If you wanna make "UNKNOWN" class 
    # criterion_bce = nn.CrossEntropyLoss(weight=torch.Tensor([1]), ignore_index=2).cuda()
    
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1])).cuda()
    # criterion_contrastive = ImagePixelwiseContrastiveLoss().cuda()

    # optim
    optimizer = select_optimizer(args.optim_function, args.initial_lr, net)

    net = net.cuda()
    
    cudnn.benchmark = True

    lr_schedule = [50, 75, 100]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=0.1)
    

    for epoch in range(lr_schedule[-1]):
        main_train(train_loader, net, criterion_bce, optimizer,
            epoch, trn_logger, trn_raw_logger, os.path.join(args.work_dir, args.exp))

        validate(valid_loader, net, criterion_bce, epoch, val_logger)

        lr_scheduler.step()

        #import ipdb; ipdb.set_trace()
        checkpoint_filename = 'model_checkpoint_{:0>3}.pth'.format(epoch + 1)
        save_checkpoint({'epoch': epoch + 1,
                            'state_dict': net.state_dict(),
                            'optimizer': optimizer.state_dict()},
                        1, work_dir,
                        checkpoint_filename)

    # test부분 여기에
    loss, f1_score, sensitivity, precision, specificity, accuracy, auc_score = test(test_loader, net, criterion_bce, epoch, tst_logger)
    print("End Phase Test Result \t")
    print("-" * 50)
    print("Final Test Loss are {} \t".format(loss)) 
    print("Final F-1 Scroe are {} \t".format(f1_score))
    print("Final Sensitivity are {} \t".format(sensitivity))
    print("Final Specificity are {} \t".format(specificity))
    print("Final Precision are {} \t".format(precision))
    print("Final Accuracy are {} \t".format(accuracy))
    print("Final AUC Score are {} \t".format(auc_score))
    print("-" * 50)
    
def main_train(trn_loader, model, criterion_bce,
                        optimizer, epoch, logger, 
                        sublogger, work_dir):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    bce_losses = AverageMeter()
    accuracies = AverageMeter()
    auces = AverageMeter()

    end = time.time()
    model.train()
    confusion_matrixs = [] 
    
    for i, (input, target) in enumerate(trn_loader):
        data_time.update(time.time() - end)
        input, target = input.float().cuda(), target.float().cuda()
        output = model(input)

        #import ipdb; ipdb.set_trace()
        target = target.unsqueeze(1)
        loss_bce = criterion_bce(output, target)

        pos_probs = torch.sigmoid(output)

        auc, acc, confusion = Performance(target.detach().cpu().numpy(), 
                                          pos_probs.detach().cpu().numpy())
        
        #import ipdb; ipdb.set_trace()

        confusion_matrixs.append(confusion)
        
        bce_losses.update((loss_bce).item(), input.size(0))
        accuracies.update(acc, input.size(0))
        auces.update(auc, input.size(0))

        optimizer.zero_grad()

        loss_bce.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        print('Training Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Classification Loss {cls_loss:.4f}({cls_losses.avg:.4f})\t'
            'Accuracy {acc:.3f} ({acces.avg:.3f})\t'
            'AUC score {auc:.3f} ({auces.avg:.3f})'.format(
            epoch, i, len(trn_loader), batch_time=batch_time,
            data_time=data_time, cls_loss=loss_bce.item(),
            cls_losses = bce_losses, acc=acc, acces=accuracies, 
            auc=auc, auces=auces))

        if i % 10 == 0 :
            sublogger.write([epoch, i, loss_bce.item(), acc, auc])

    
    final_confusion = np.sum(confusion_matrixs, axis=0)
    tn, fp, fn, tp = final_confusion    
    sensitivity = tp / (tp+fn)
    precision = tp / (tp+fp)
    
    f1_score = 2 * ((sensitivity*precision) / (sensitivity+precision))
    specificity = tn / (tn+fp)
    accuracy = (tp+tn) / (tp+fp+tn+fn)
    auc_score = auces.avg
    
    logger.write([epoch, bce_losses.avg, f1_score, sensitivity, precision, specificity, accuracy, auc_score])

def validate(val_loader, model, criterion_dice, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    bce_losses = AverageMeter()
    accuracies = AverageMeter()
    auces = AverageMeter()
    confusion_matrixs = [] 
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input, target = input.float().cuda(), target.float().cuda()

            output = model(input)
            target = target.unsqueeze(1)
            
            loss_bce = criterion_dice(output, target)
            pos_probs = torch.sigmoid(output)


            auc, acc, confusion = Performance(target.detach().cpu().numpy(), 
                                          pos_probs.detach().cpu().numpy())
            

            confusion_matrixs.append(confusion)
            
            bce_losses.update((loss_bce).item(), input.size(0))
            accuracies.update(acc, input.size(0))
            auces.update(auc, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            print('Validation Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Classification Loss {cls_loss:.4f}({cls_losses.avg:.4f})\t'
            'Accuracy {acc:.3f} ({acces.avg:.3f})\t'
            'AUC score {auc:.3f} ({auces.avg:.3f})'.format(
            epoch, i, len(val_loader), batch_time=batch_time,
            data_time=data_time, cls_loss=loss_bce.item(),
            cls_losses = bce_losses, acc=acc, acces=accuracies, 
            auc=auc, auces=auces))

    final_confusion = np.sum(confusion_matrixs, axis=0)
    tn, fp, fn, tp = final_confusion    
    sensitivity = tp / (tp+fn)
    precision = tp / (tp+fp)
    
    f1_score = 2 * ((sensitivity*precision) / (sensitivity+precision))
    specificity = tn / (tn+fp)
    accuracy = (tp+tn) / (tp+fp+tn+fn)
    auc_score = auces.avg
    
    logger.write([epoch, bce_losses.avg, f1_score, sensitivity, 
                  precision, specificity, accuracy, auc_score])
    

def test(tst_loader, model, criterion_dice, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    bce_losses = AverageMeter()
    accuracies = AverageMeter()
    auces = AverageMeter()
    confusion_matrixs = [] 
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(tst_loader):

            input, target = input.cuda(), target.cuda()
            target = target.unsqueeze(1)
            output = model(input)

            loss_bce = criterion_dice(output, target)
            pos_probs = torch.sigmoid(output)


            auc, acc, confusion = Performance(target.detach().cpu().numpy(), 
                                          pos_probs.detach().cpu().numpy())

            confusion_matrixs.append(confusion)
            
            bce_losses.update((loss_bce).item(), input.size(0))
            accuracies.update(acc, input.size(0))
            auces.update(auc, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            
            print('Final Test: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Classification Loss {cls_loss:.4f}({cls_losses.avg:.4f})\t'
            'Accuracy {acc:.3f} ({acces.avg:.3f})\t'
            'AUC score {auc:.3f} ({auces.avg:.3f})'.format(
            epoch, i, len(tst_loader), batch_time=batch_time,
            data_time=data_time, cls_loss=loss_bce.item(),
            cls_losses = bce_losses, acc=acc, acces=accuracies, 
            auc=auc, auces=auces))

    final_confusion = np.sum(confusion_matrixs, axis=0)
    tn, fp, fn, tp = final_confusion
    
    sensitivity = tp / (tp+fn)
    precision = tp / (tp+fp)
    
    f1_score = 2 * ((sensitivity*precision) / (sensitivity+precision))
    
    specificity = tn / (tn+fp)
    accuracy = (tp+tn) / (tp+fp+tn+fn)
    auc_score = auces.avg
    
    logger.write([epoch, bce_losses.avg, f1_score, sensitivity, 
                  precision, specificity, accuracy, auc_score])
    
    return bce_losses.avg, f1_score, sensitivity, precision, specificity, accuracy, auc_score



if __name__=='__main__':
    main()