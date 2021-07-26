#
import os
import random
import datetime
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.data_provider_city import SingleLoader

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from rgb_options import opt
from model.dbsn import DBSN_Model
from loss.losses import L2Loss
from utils.utils import batch_psnr,findLastCheckpoint

seed=0
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if seed == 0:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # Init loggers
    os.makedirs(args.log_dir, exist_ok=True)
    if args.noise_type == 'gaussian':
        if len(args.train_noiseL)>1:
            noise_level = 'nL'+str(int(args.train_noiseL[0]))+','+str(int(args.train_noiseL[1]))
        else:
            noise_level = 'nL'+str(int(args.train_noiseL[0]))
    elif args.noise_type == 'poisson_gaussian':
        if len(args.train_noiseL)>2:
            noise_level = 'sigmaS'+str(int(args.train_noiseL[1]))+'_sigmaC'+str(int(args.train_noiseL[2]))
        else:
            noise_level = 'sigmaS'+str(int(args.train_noiseL[0]))+'_sigmaC'+str(int(args.train_noiseL[1]))
    elif args.noise_type == 'multivariate_gaussian':
        noise_level = 'fix'
    else:
        raise ValueError('Noise_type [%s] is not found.' % (args.noise_type))    
    checkpoint_dir = args.save_prefix + '_' + noise_level
    ckpt_save_path = os.path.join(args.log_dir, checkpoint_dir)
    os.makedirs(ckpt_save_path, exist_ok=True)
    logger_fname = os.path.join(ckpt_save_path, checkpoint_dir+'_log.txt')

    with open(logger_fname, "w") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Log output (%s) ================\n' % now) 
        log_file.write('Parameters \n') 
        for key in opt.__dict__:
            p = key+': '+str(opt.__dict__[key])
            log_file.write('%s\n' % p)
        
    # net architecture
    dbsn_net = DBSN_Model(in_ch = args.input_channel,
                            out_ch = args.output_channel,
                            mid_ch = args.middle_channel,
                            blindspot_conv_type = args.blindspot_conv_type,
                            blindspot_conv_bias = args.blindspot_conv_bias,
                            br1_block_num = args.br1_block_num,
                            br1_blindspot_conv_ks =args.br1_blindspot_conv_ks,
                            br2_block_num = args.br2_block_num,
                            br2_blindspot_conv_ks = args.br2_blindspot_conv_ks,
                            activate_fun = args.activate_fun)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loss function
    criterion = L2Loss().to(device)

    # Move to GPU
    # dbsn_model = nn.DataParallel(dbsn_net, args.device_ids).cuda()
    dbsn_model = dbsn_net.to(device)
    # new or continue
    ckpt_save_prefix = args.save_prefix + '_ckpt_e'
    initial_epoch = findLastCheckpoint(save_dir=ckpt_save_path, save_pre = ckpt_save_prefix)
    if initial_epoch > 0:
        print('*****resuming by loading epoch %03d' % initial_epoch)
        args.resume = "continue"
        args.last_ckpt = os.path.join(ckpt_save_path, ckpt_save_prefix + str(initial_epoch) + '.pth')

    # Optimizer
    training_params = None
    optimizer_dbsn = None
    if args.resume == "continue":
        tmp_ckpt=torch.load(args.last_ckpt,map_location=torch.device('cuda', args.device_ids[0]))
        training_params = tmp_ckpt['training_params']
        start_epoch = training_params['start_epoch']
        # Initialize dbsn_model
        pretrained_dict=tmp_ckpt['state_dict_dbsn']
        model_dict=dbsn_model.state_dict()
        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert(len(pretrained_dict)==len(pretrained_dict_update))
        assert(len(pretrained_dict_update)==len(model_dict))
        model_dict.update(pretrained_dict_update)
        dbsn_model.load_state_dict(model_dict)
        optimizer_dbsn = optim.Adam(dbsn_model.parameters(), lr=args.lr_dbsn)
        optimizer_dbsn.load_state_dict(tmp_ckpt['optimizer_state_dbsn'])
        schedule_dbsn = torch.optim.lr_scheduler.MultiStepLR(optimizer_dbsn, milestones=args.steps, gamma=args.decay_rate)

    elif args.resume == "new":
        training_params = {}
        training_params['step'] = 0
        start_epoch = 0
        # Initialize dbsn_model
        optimizer_dbsn = optim.Adam(dbsn_model.parameters(), lr=args.lr_dbsn)
        schedule_dbsn = torch.optim.lr_scheduler.MultiStepLR(optimizer_dbsn, milestones=args.steps, gamma=args.decay_rate)


    # logging
    dbsn_params = sum(param.numel() for param in dbsn_net.parameters() if param.requires_grad)/(1e6)
    with open(logger_fname, "a") as log_file:
        log_file.write('model created\n')
        log_file.write('DBSN trainable parameters: {dbsn_params:.2f}M\n'.format(dbsn_params=dbsn_params))


    # set training set
    # SingleLoader

    data_set = SingleLoader(noise_dir=args.noise_dir, gt_dir=args.gt_dir, image_size=args.image_size)

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # logging
    # with open(logger_fname, "a") as log_file:
    #     log_file.write('training/val dataset created\n')
    #     log_file.write('number of training examples {0} \n'.format(dataset_num))


    # --------------------------------------------
    # Checkpoint
    # --------------------------------------------
    if args.resume == 'continue':
        # # evaluating the loaded model first ...
        # print("Starting from epoch: %d "%(start_epoch))
        # logging
        with open(logger_fname, "a") as log_file:
            log_file.write('checkpoint evaluation on epoch {0} ... \n'.format(start_epoch))
        dbsn_model.eval()
        with torch.no_grad():
            psnr_val = 0
            for count, (img_val,img_noise_val) in enumerate(data_loader):
                # prepare data
                # img_val = data['clean'].cuda()
                # img_noise_val = data['noisy'].cuda()
                img_val = img_val.to(device)
                img_noise_val = img_noise_val.to(device)

                batch,C,H,W = img_noise_val.shape
                # forward
                mu_out_val, _ = dbsn_model(img_noise_val)
                # compute PSNR
                psnr = batch_psnr(mu_out_val.clamp(0., 1.), img_val.clamp(0., 1.), 1.)
                psnr_val+=psnr
                # # print
                # print("Image: %d, psnr_mu: %.4f" % (count, psnr))
                # logging
                with open(logger_fname, "a") as log_file:
                    log_file.write('Image {0} \t' 
                        'psnr_mu:{psnr:.4f} \n'.format(count, psnr=psnr))
            psnr_val /= len(data_loader)
            # # print
            # print("Checkpoint avg psnr_mu: %.4f \n" % (psnr_val))
            # logging
            with open(logger_fname, "a") as log_file:
                log_file.write('checkpoint avg psnr_mu:{psnr_val:.4f} \n\n'.format(psnr_val=psnr_val))
            val_psnr_pre = psnr_val
            idx_epoch = start_epoch


    # --------------------------------------------
    # Training
    # --------------------------------------------
    # logging
    with open(logger_fname, "a") as log_file:
        log_file.write('started training\n')            
    val_psnr_curr = 0
    for epoch in range(start_epoch, args.epoch):
        epoch_start_time = time.time()
        # print
        print('lr: %.10f' % (optimizer_dbsn.state_dict()["param_groups"][0]["lr"]))
        # logging
        with open(logger_fname, "a") as log_file:
            log_file.write('dbsn_lr: {lr:.10f} \n'.format(lr=optimizer_dbsn.state_dict()["param_groups"][0]["lr"]))
        # begin training
        dbsn_model.train()
        for i, (img_train,img_noise) in enumerate(data_loader):
            # load data
            # img_train = data['clean'].cuda()
            # img_noise = data['noisy'].cuda()
            img_train = img_train.to(device)
            img_noise = img_noise.to(device)

            batch,C,H,W = img_noise.shape
            #
            optimizer_dbsn.zero_grad()
            # forward
            mu_out, _ = dbsn_model(img_noise)
            #
            loss = criterion(img_noise, mu_out)
            loss = loss / (2*args.batch_size)
            loss_value = loss.item()
            # backward
            loss.backward()
            optimizer_dbsn.step()
            # Results
            training_params['step'] += args.batch_size
            # print results
            if training_params['step'] % (args.batch_size*args.display_freq) == 0:
                with torch.no_grad():
                    # compute training psnr 
                    train_psnr = batch_psnr(mu_out.clamp(0., 1.), img_train.clamp(0., 1.), 1.0)
                    # print
                    print(("[epoch %d/%d][%d/], loss:%.4f, psnr_mu: %.4f") %
                    (epoch, args.epoch, i*args.batch_size, loss_value, train_psnr))
                    # logging
                    # with open(logger_fname, "a") as log_file:
                    #     log_file.write('Epoch: [{0}/{1}][{2}/{3}] \t'
                    #         'loss {loss_value: .4f} \t'
                    #         'psnr_mu:{train_psnr:.4f} \n'.format(epoch, args.epoch, i*args.batch_size, len(dataset),
                    #         loss_value=loss_value, train_psnr=train_psnr))

        schedule_dbsn.step()
        # taking time for each epoch
        tr_take_time = time.time() - epoch_start_time


        # --------------------------------------------
        # validation
        # --------------------------------------------
        # # print
        # print("Evaluating on "+str(val_setname[0]))
        # logging
        # with open(logger_fname, "a") as log_file:
        #     log_file.write('Evaluation on %s \n' % (str(val_setname[0])) )
        #
        dbsn_model.eval()
        val_start_time = time.time()
        with torch.no_grad():
            psnr_val = 0
            for count, (img_val,img_noise_val) in enumerate(data_loader):
                #
                # img_val = data['clean'].cuda()
                # img_noise_val = data['noisy'].cuda()
                img_val = img_val.to(device)
                img_noise_val = img_noise_val.to(device)

                batch,C,H,W = img_noise_val.shape
                # forward
                mu_out_val, _ = dbsn_model(img_noise_val)
                # compute PSNR
                psnr = batch_psnr(mu_out_val.clamp(0., 1.), img_val.clamp(0., 1.), 1.)
                psnr_val+=psnr
                # print
                # print("Image: %d, psnr_mu: %.4f " % (count, psnr))
                # logging
                # with open(logger_fname, "a") as log_file:
                #     log_file.write('Image {0} \t'
                #         'psnr_mu:{psnr:.4f} \n'.format(count, psnr=psnr))
            torch.cuda.synchronize()
            val_take_time = time.time() - val_start_time
            psnr_val /= len(data_loader)

            val_psnr_curr = psnr_val
            if epoch==0:
                val_psnr_pre = val_psnr_curr

        # save model and checkpoint
        if val_psnr_curr >= val_psnr_pre or (epoch+1) % args.save_model_freq == 0:
            training_params['start_epoch'] = epoch
            save_dict = {'state_dict_dbsn': dbsn_model.state_dict(),
                        'optimizer_state_dbsn': optimizer_dbsn.state_dict(),
                        'schedule_state_dbsn': schedule_dbsn.state_dict(),
                        'training_params': training_params,
                        'args': args}
            torch.save(save_dict, os.path.join(ckpt_save_path, ckpt_save_prefix + '{}.pth'.format(epoch)))
            if val_psnr_curr >= val_psnr_pre:
                val_psnr_pre = val_psnr_curr
                idx_epoch = epoch
                torch.save(dbsn_model.state_dict(), os.path.join(ckpt_save_path, 'dbsn_net_best_e{}.pth'.format(epoch)))
            del save_dict
        # print 
        print('RGB_Pretrain_mu Epoch [%d/%d]:' % (epoch, args.epoch))
        print('val psnr_mu: %.4f \t val Train_time: %d sec \t Val_time: %d sec \t' % (psnr_val, tr_take_time, val_take_time))
        print('Best Val PSNR_mu=%.4f with Epoch %d \n' % (val_psnr_pre, idx_epoch))
        # logging
        # with open(logger_fname, "a") as log_file:
        #     log_file.write('Epoch: [{0}/{1}/{2}] \t'
        #     'Train_time:{tr_take_time:.1f} sec \t'
        #     'Val_time:{val_take_time:.1f} sec \t'
        #     'VAL psnr_mu:{psnr_val:.4f} \t'
        #     'Best VAL psnr_mu:{best_psnr_val:.4f} \n'.format(epoch, args.epoch, idx_epoch,
        #     tr_take_time=tr_take_time, val_take_time=val_take_time,
        #     psnr_val=psnr_val, best_psnr_val=val_psnr_pre))


if __name__ == "__main__":

    main(opt)

    exit(0)





