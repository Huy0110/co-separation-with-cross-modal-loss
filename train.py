#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from scipy.misc import imsave
import scipy.io.wavfile as wavfile
import numpy as np
import torch
from torch.autograd import Variable
import librosa
from utils import utils,viz
from models import criterion
import torch.nn.functional as F
import random

def create_optimizer(nets, opt):
        (net_visual, net_unet, net_classifier, net_vocal, net_facial_attribtes) = nets
        param_groups = [{'params': net_visual.parameters(), 'lr': opt.lr_visual},
                        {'params': net_unet.parameters(), 'lr': opt.lr_unet},
                        {'params': net_classifier.parameters(), 'lr': opt.lr_classifier},
                        {'params': net_vocal.parameters(), 'lr': opt.lr_vocal_attributes},
                        {'params': net_facial_attribtes.parameters(), 'lr': opt.lr_facial_attributes}]
        if opt.optimizer == 'sgd':
            return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adam':
            return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

def print_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

def save_visualization(vis_rows, outputs, batch_data, save_dir, opt):
    # fetch data and predictions
    mag_mix = batch_data['audio_mix_mags']
    phase_mix = batch_data['audio_mix_phases']
    visuals = batch_data['visuals']

    pred_masks_ = outputs['pred_mask']
    gt_masks_ = outputs['gt_mask']
    mag_mix_ = outputs['audio_mix_mags']
    weight_ = outputs['weight']
    visual_object = outputs['visual_object']
    gt_label = outputs['gt_label']
    _, pred_label = torch.max(output['pred_label'], 1)
    label_list = ['Banjo', 'Cello', 'Drum', 'Guitar', 'Harp', 'Harmonica', 'Oboe', 'Piano', 'Saxophone', \
                    'Trombone', 'Trumpet', 'Violin', 'Flute','Accordion', 'Horn']

    # unwarp log scale
    B = mag_mix.size(0)
    if opt.log_freq:
        grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame//2+1, gt_masks_.size(3), warp=False)).to(opt.device)
        pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
        gt_masks_linear = F.grid_sample(gt_masks_, grid_unwarp)
    else:
        pred_masks_linear = pred_masks_
        gt_masks_linear = gt_masks_

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    pred_masks_ = pred_masks_.detach().cpu().numpy()
    pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
    gt_masks_ = gt_masks_.detach().cpu().numpy()
    gt_masks_linear = gt_masks_linear.detach().cpu().numpy()
    visual_object = visual_object.detach().cpu().numpy()
    gt_label = gt_label.detach().cpu().numpy()
    pred_label = pred_label.detach().cpu().numpy()

    # loop over each example
    for j in range(min(B, opt.num_visualization_examples)):
        row_elements = []

        # video names
        prefix = str(j) + '-' + label_list[int(gt_label[j])] + '-' + label_list[int(pred_label[j])]
        utils.mkdirs(os.path.join(save_dir, prefix))

        # save mixture
        mix_wav = utils.istft_coseparation(mag_mix[j, 0], phase_mix[j, 0], hop_length=opt.stft_hop)
        mix_amp = utils.magnitude2heatmap(mag_mix_[j, 0])
        weight = utils.magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imsave(os.path.join(save_dir, filename_mixmag), mix_amp[::-1, :, :])
        imsave(os.path.join(save_dir, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(save_dir, filename_mixwav), opt.audio_sampling_rate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # GT and predicted audio reconstruction
        gt_mag = mag_mix[j, 0] * gt_masks_linear[j, 0]
        gt_wav = utils.istft_coseparation(gt_mag, phase_mix[j, 0], hop_length=opt.stft_hop)
        pred_mag = mag_mix[j, 0] * pred_masks_linear[j, 0]
        preds_wav = utils.istft_coseparation(pred_mag, phase_mix[j, 0], hop_length=opt.stft_hop)

        # output masks
        filename_gtmask = os.path.join(prefix, 'gtmask.jpg')
        filename_predmask = os.path.join(prefix, 'predmask.jpg')
        gt_mask = (np.clip(gt_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        pred_mask = (np.clip(pred_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        imsave(os.path.join(save_dir, filename_gtmask), gt_mask[::-1, :])
        imsave(os.path.join(save_dir, filename_predmask), pred_mask[::-1, :])

        # ouput spectrogram (log of magnitude, show colormap)
        filename_gtmag = os.path.join(prefix, 'gtamp.jpg')
        filename_predmag = os.path.join(prefix, 'predamp.jpg')
        gt_mag = utils.magnitude2heatmap(gt_mag)
        pred_mag = utils.magnitude2heatmap(pred_mag)
        imsave(os.path.join(save_dir, filename_gtmag), gt_mag[::-1, :, :])
        imsave(os.path.join(save_dir, filename_predmag), pred_mag[::-1, :, :])

        # output audio
        filename_gtwav = os.path.join(prefix, 'gt.wav')
        filename_predwav = os.path.join(prefix, 'pred.wav')
        wavfile.write(os.path.join(save_dir, filename_gtwav), opt.audio_sampling_rate, gt_wav)
        wavfile.write(os.path.join(save_dir, filename_predwav), opt.audio_sampling_rate, preds_wav)

        row_elements += [
                {'image': filename_predmag, 'audio': filename_predwav},
                {'image': filename_gtmag, 'audio': filename_gtwav},
                {'image': filename_predmask},
                {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)

#used to display validation loss
def display_val(model, crit, writer, index, dataset_val, opt):
        # remove previous viz results
        save_dir = os.path.join('.', opt.checkpoints_dir, opt.name, 'visualization')
        utils.mkdirs(save_dir)

        #initial results lists
        accuracies = []
        classifier_losses = []
        coseparation_losses = []
        crossmodal_losses = []

        # initialize HTML header
        visualizer = viz.HTMLVisualizer(os.path.join(save_dir, 'index.html'))
        header = ['Filename', 'Input Mixed Audio']
        header += ['Predicted Audio' 'GroundTruth Audio', 'Predicted Mask','GroundTruth Mask', 'Loss weighting']
        visualizer.add_header(header)
        vis_rows = []

        with torch.no_grad():
            for i, val_data in enumerate(dataset_val):
                if i < opt.validation_batches:
                    output = model.forward(val_data)
                    loss_classification = crit['loss_classification']
                    classifier_loss = loss_classification(output['pred_label'], Variable(output['gt_label'], requires_grad=False)) * opt.classifier_loss_weight
                    coseparation_loss = get_coseparation_loss(output, opt, crit['loss_coseparation']) * opt.coseparation_loss_weight
                    crossmodal_loss = get_crossmodal_loss(output, opt, crit['loss_triplet']) * opt.crossmodal_loss_weight
                    classifier_losses.append(classifier_loss.item()) 
                    coseparation_losses.append(coseparation_loss.item())
                    crossmodal_losses.append(crossmodal_loss.item())

                    gt_label = output['gt_label']
                    _, pred_label = torch.max(output['pred_label'], 1)
                    accuracy = torch.sum(gt_label == pred_label).item() * 1.0 / pred_label.shape[0]
                    accuracies.append(accuracy)
                else:
                    if opt.validation_visualization:
                        output = model.forward(val_data)
                        save_visualization(vis_rows, output, val_data, save_dir, opt) #visualize one batch
                    break

        avg_accuracy = sum(accuracies)/len(accuracies)
        avg_classifier_loss = sum(classifier_losses)/len(classifier_losses)
        avg_coseparation_loss = sum(coseparation_losses)/len(coseparation_losses)
        avg_crossmodal_loss = sum(crossmodal_losses)/len(crossmodal_losses)
        if opt.tensorboard:
            writer.add_scalar('data/val_classifier_loss', avg_classifier_loss, index)
            writer.add_scalar('data/val_accuracy', avg_accuracy, index)
            writer.add_scalar('data/val_coseparation_loss', avg_coseparation_loss, index)
            writer.add_scalar('data/val_crossmodal_loss', avg_crossmodal_loss, index)
        print('val accuracy: %.3f' % avg_accuracy)
        print('val classifier loss: %.3f' % avg_classifier_loss)
        print('val coseparation loss: %.3f' % avg_coseparation_loss)
        print('val crossmodal loss: %.5f' % avg_crossmodal_loss)
        return avg_coseparation_loss + avg_classifier_loss + avg_crossmodal_loss

def get_coseparation_loss(output, opt, loss_coseparation):
        #initialize a dic to store the index of the list
        vid_index_dic ={}
        vids = output['vids'].squeeze(1).cpu().numpy()
        O = vids.shape[0]
        count = 0
        for i in range(O):
            if vids[i] not in vid_index_dic:
                vid_index_dic[vids[i]] = count
                count = count + 1

        #initialize three lists of length = number of video clips to reconstruct
        predicted_mask_list = [None for i in range(len(vid_index_dic.keys()))]
        gt_mask_list = [None for i in range(len(vid_index_dic.keys()))]
        weight_list = [None for i in range(len(vid_index_dic.keys()))]

        #iterate through all objects
        gt_masks = output['gt_mask']
        mask_prediction = output['pred_mask']
        weight = output['weight']

        # print(gt_masks)
        # print(mask_prediction)
        # print(weight)
        for i in range(O):
            if predicted_mask_list[vid_index_dic[vids[i]]] is None:
                gt_mask_list[vid_index_dic[vids[i]]] = gt_masks[i,:,:,:]
                weight_list[vid_index_dic[vids[i]]] = weight[i,:,:,:]
                # weight_list[vid_index_dic[vids[i]]] = gt_masks[i,:,:,:]
                predicted_mask_list[vid_index_dic[vids[i]]] = mask_prediction[i,:,:,:]
            else:
                predicted_mask_list[vid_index_dic[vids[i]]] = predicted_mask_list[vid_index_dic[vids[i]]] + mask_prediction[i,:,:,:]

        if opt.mask_loss_type == 'BCE':
            for i in range(O):
                #clip the prediction results to make it in the range of [0,1] for BCE loss
                predicted_mask_list[vid_index_dic[vids[i]]] = torch.clamp(predicted_mask_list[vid_index_dic[vids[i]]], 0, 1)
        # print("len predict mask",len(predicted_mask_list))
        # print("len gt mask",len(gt_mask_list))
        coseparation_loss = loss_coseparation(predicted_mask_list, gt_mask_list, weight_list)
        #print(type(coseparation_loss))
        return coseparation_loss

def get_crossmodal_loss1(output, opt, loss_triplet):
    visual_feature = output['visual_embadding']
    audio_embaddings = output['audio_embeddings_gt']
    audio_embaddings_pred = output[' audio_embeddings_pred']

    if random.random() > 0.5:
        audio_embaddings = audio_embaddings_pred
    else:
        audio_embaddings = audio_embaddings


    #crossmodal_loss = loss_triplet(audio_embeddings_A1, identity_feature_A, identity_feature_B) + loss_triplet(audio_embeddings_A2, identity_feature_A, identity_feature_B) + loss_triplet(audio_embeddings_B1, identity_feature_B, identity_feature_A) + loss_triplet(audio_embeddings_B2, identity_feature_B, identity_feature_A)
    crossmodal_loss = 0
    for i in range(1):
        if i == 1 : 
            crossmodal_loss = crossmodal_loss + loss_triplet(audio_embaddings[i], visual_feature[i], visual_feature[0]) + loss_triplet(audio_embaddings[0], visual_feature[0], visual_feature[i])
        crossmodal_loss = crossmodal_loss + loss_triplet(audio_embaddings[i], visual_feature[i], visual_feature[i+1]) + loss_triplet(audio_embaddings[i+1], visual_feature[i+1], visual_feature[i])
    return crossmodal_loss

#https://arxiv.org/pdf/2101.03149.pdf
#https://github.com/facebookresearch/VisualVoice
def get_crossmodal_loss(output, opt, loss_triplet):
    visual_feature = output['visual_embadding']
    audio_embaddings = output['audio_embeddings_gt']
    audio_embaddings_pred = output['audio_embeddings_pred']

    #audio_embaddings_pred = F.normalize(audio_embaddings_pred, p=2, dim=1)
    #audio_embaddings = F.normalize(audio_embaddings, p=2, dim=1)
    #visual_feature = F.normalize(visual_feature, p=2, dim=1)
    vids = output['vids']
    count = 0.0
    #print(vids)

    if random.random() > 0.5:
        #print("1")
        audio_embaddings = audio_embaddings_pred
    else:
        audio_embaddings = audio_embaddings
       #print("0")

    #gt_labels = output['gt_labels']
    #print(audio_embaddings.shape)
    #print(visual_feature.shape)

    for i in range(visual_feature.shape[0]):
        if np.count_nonzero(vids.cpu().detach().numpy() == vids[i].cpu().detach().numpy()) > 1:
            #print("Trung o i :")
            #print(vids[i])
            continue
        for j in range(visual_feature.shape[0]):
            if np.count_nonzero(vids.cpu().detach().numpy() == vids[j].cpu().detach().numpy()) > 1:
                #print("Trung o j :")
                #print(vids[j])
                continue
            if (audio_embaddings[i].cpu().detach().numpy() != audio_embaddings[j].cpu().detach().numpy()).any():
                #print("Duoc xet: ")
                #print(vids[i])
                #print(vids[j])
                #print("------------")
                #crossmodal_loss = crossmodal_loss + loss_triplet(audio_embaddings[i], visual_feature[i], visual_feature[j]) + loss_triplet(audio_embaddings[j], visual_feature[j], visual_feature[i])
                if count == 0.0 :
                    crossmodal_loss = loss_triplet(audio_embaddings[i], visual_feature[i], visual_feature[j])
                    count = count + 1.0
                else :
                    crossmodal_loss = crossmodal_loss + loss_triplet(audio_embaddings[i], visual_feature[i], visual_feature[j])
                    count = count + 1.0
                #print(count)
                #print(loss_triplet(audio_embaddings[i], visual_feature[i], visual_feature[j]))
                #print("--------------------------------")
                
                #print("---------------------")
                #print("Audio embaddings: ")
                #print(audio_embaddings[i].shape)
                #print("Visual feature:")
                #print(visual_feature[i].shape)
                #print(visual_feature[j].shape)
                #print("----------------------")
    if count == 0.0:
        print("All duet")
        for i in range(visual_feature.shape[0]):
            for j in range(visual_feature.shape[0]):
                if (audio_embaddings[i].cpu().detach().numpy() != audio_embaddings[j].cpu().detach().numpy()).any():
                    if count == 0.0 :
                        crossmodal_loss = loss_triplet(audio_embaddings[i], visual_feature[i], visual_feature[j])
                        count = count + 1.0
                    else :
                        crossmodal_loss = crossmodal_loss + loss_triplet(audio_embaddings[i], visual_feature[i], visual_feature[j])
                        count = count + 1.0
        crossmodal_loss = crossmodal_loss / count * 2.0
    else:
        crossmodal_loss = crossmodal_loss / count * 2.0
    #print(type(crossmodal_loss))
    return crossmodal_loss


#parse arguments
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

if opt.with_additional_scene_image:
    opt.number_of_classes = opt.number_of_classes + 1

#construct data loader
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

#create validation set data loader if validation_on option is set
if opt.validation_on:
        #temperally set to val to load val data
        opt.mode = 'val'
        data_loader_val = CreateDataLoader(opt)
        dataset_val = data_loader_val.load_data()
        dataset_size_val = len(data_loader_val)
        print('#validation images = %d' % dataset_size_val)
        opt.mode = 'train' #set it back

if opt.tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    writer = None

# Network Builders
builder = ModelBuilder()
#if identity feature dim is not 512, for resnet reduce dimension to this feature dim
'''
if opt.identity_feature_dim = 512:
    opt.with_fc = True
else:
    opt.with_fc = False
'''
net_visual = builder.build_visual(
        pool_type=opt.visual_pool,
        fc_out = 512,
        weights=opt.weights_visual)  
net_unet = builder.build_unet(
        unet_num_layers = opt.unet_num_layers,
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        weights=opt.weights_unet)
net_classifier = builder.build_classifier(
        pool_type=opt.classifier_pool,
        num_of_classes=opt.number_of_classes,
        input_channel=opt.unet_output_nc,
        weights=opt.weights_classifier)
net_vocal = builder.build_vocal(
        pool_type=opt.audio_pool,
        input_channel=1,
        with_fc= True,
        fc_out = opt.identity_feature_dim,
        weights=opt.weights_vocal)
net_facial_attribtes = builder.build_facial(
        pool_type=opt.visual_pool,
        fc_out = opt.identity_feature_dim,
        with_fc=True,
        weights=opt.weights_facial)
nets = (net_visual, net_unet, net_classifier, net_vocal, net_facial_attribtes)
 
# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=[2]) #opt.gpu_ids)
#model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)

# Set up optimizer
optimizer = create_optimizer(nets, opt)

# Set up loss functions
if opt.triplet_loss_type == 'tripletCosine':
    loss_triplet = criterion.TripletLossCosine(opt.margin)
elif opt.triplet_loss_type == 'triplet':
    loss_triplet = criterion.TripletLoss(opt.margin)
loss_classification = criterion.CELoss()
if opt.mask_loss_type == 'L1':
    loss_coseparation = criterion.L1Loss()
elif opt.mask_loss_type == 'L2':
    loss_coseparation = criterion.L2Loss()
elif opt.mask_loss_type == 'BCE':
    loss_coseparation = criterion.BCELoss()
if(len(opt.gpu_ids) > 0):
    loss_triplet.cuda(opt.gpu_ids[0])
    loss_classification.cuda(opt.gpu_ids[0])
    loss_coseparation.cuda(opt.gpu_ids[0])
crit = {'loss_classification': loss_classification, 'loss_coseparation': loss_coseparation, 'loss_triplet': loss_triplet}


#initialization
total_batches = 0
data_loading_time = []
model_forward_time = []
model_backward_time = []
batch_classifier_loss = []
batch_coseparation_loss = []
batch_crossmodal_loss = []
best_err = float("inf")

for epoch in range(1 + opt.epoch_count, opt.niter+1):
        torch.cuda.synchronize()
        epoch_start_time = time.time()

        if(opt.measure_time):
                iter_start_time = time.time()
        for i, data in enumerate(dataset):
                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_loaded_time = time.time()

                #print(data['label'].size())
                total_batches += 1

                #forward pass
                model.zero_grad()
                #print("0")
                output = model.forward(data)
                #print("1")

                #compute loss
                #classifier_loss
                classifier_loss = loss_classification(output['pred_label'], Variable(output['gt_label'], requires_grad=False)) * opt.classifier_loss_weight

                #coseparation loss
                coseparation_loss = get_coseparation_loss(output, opt, loss_coseparation) * opt.coseparation_loss_weight

                #crossmodal loss
                crossmodal_loss = get_crossmodal_loss(output, opt, loss_triplet) * opt.crossmodal_loss_weight

                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_data_forwarded_time = time.time()
                #store losses for this batch
                batch_classifier_loss.append(classifier_loss.item())
                batch_coseparation_loss.append(coseparation_loss.item())
                batch_crossmodal_loss.append(crossmodal_loss.item())

                optimizer.zero_grad()
                classifier_loss.backward(retain_graph=True)
                
                coseparation_loss.backward(retain_graph=True)
                crossmodal_loss.backward()
                optimizer.step()

                if(opt.measure_time):
                    torch.cuda.synchronize()
                    iter_model_backwarded_time = time.time()

                if(opt.measure_time):
                        torch.cuda.synchronize()
                        iter_model_backwarded_time = time.time()
                        data_loading_time.append(iter_data_loaded_time - iter_start_time)
                        model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                        model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

                if(total_batches % opt.display_freq == 0):
                        print('Display training progress at (epoch %d, total_batches %d)' % (epoch, total_batches))
                        avg_classifier_loss = sum(batch_classifier_loss)/len(batch_classifier_loss)
                        avg_coseparation_loss = sum(batch_coseparation_loss)/len(batch_coseparation_loss)
                        avg_crossmodal_loss = sum(batch_crossmodal_loss)/len(batch_crossmodal_loss)

                        print('classifier loss: %.3f, co-separation loss: %.3f, cross_modal loss: %.3f' \
                            % (avg_classifier_loss, avg_coseparation_loss, avg_crossmodal_loss))
                        batch_classifier_loss = []
                        batch_coseparation_loss = []
                        batch_crossmodal_loss = []
                        if opt.tensorboard:
                            writer.add_scalar('data/classifier_loss', avg_classifier_loss, i)
                            writer.add_scalar('data/coseparation_loss', avg_coseparation_loss, i)
                            writer.add_scalar('data/crossmodal_loss', avg_crossmodal_loss, i)

                        if(opt.measure_time):
                                print('average data loading time: %.3f' % (sum(data_loading_time)/len(data_loading_time)))
                                print('average forward time: %.3f' % (sum(model_forward_time)/len(model_forward_time)))
                                print('average backward time: %.3f' % (sum(model_backward_time)/len(model_backward_time)))
                                data_loading_time = []
                                model_forward_time = []
                                model_backward_time = []
                        print('end of display \n')

                if(total_batches % opt.save_latest_freq == 0):
                        print('saving the latest model (epoch %d, total_batches %d)' % (epoch, total_batches))
                        torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_latest.pth'))
                        torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'unet_latest.pth'))
                        torch.save(net_classifier.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'classifier_latest.pth'))
                        torch.save(net_vocal.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'vocal_latest.pth'))
                        torch.save(net_facial_attribtes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'facial_latest.pth'))
                        print('Latest learning rate:')
                        print_learning_rate(optimizer)
                if(total_batches % opt.validation_freq == 0 and opt.validation_on):
                        model.eval()
                        opt.mode = 'val'
                        print('Display validation results at (epoch %d, total_batches %d)' % (epoch, total_batches))
                        val_err = display_val(model, crit, writer, total_batches, dataset_val, opt)
                        print('end of display \n')
                        model.train()
                        opt.mode = 'main'
                        #save the model that achieves the smallest validation error
                        if val_err < best_err:
                            best_err = val_err
                            print('saving the best model (epoch %d, total_batches %d) with validation error %.3f\n' % (epoch, total_batches, val_err))
                            torch.save(net_visual.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'visual_best.pth'))
                            torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'unet_best.pth'))
                            torch.save(net_classifier.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'classifier_best.pth'))
                            torch.save(net_vocal.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'vocal_best.pth'))
                            torch.save(net_facial_attribtes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'facial_best.pth'))
                #decrease learning rate
                if(total_batches in opt.lr_steps):
                        decrease_learning_rate(optimizer, opt.decay_factor)
                        print('decreased learning rate by ', opt.decay_factor)

                if(opt.measure_time):
                        torch.cuda.synchronize()
                        iter_start_time = time.time()
        opt.mode = 'train'
