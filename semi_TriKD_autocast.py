import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import yaml

from dataset.semi import SemiDataset_CPS
from model.semseg.model_helper_kd import TinyViTUperBUilder, ResUperBuilder, ViTUperBuilder
from supervised_tiny import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

# Using autocast and GradScaler
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler

scaler = GradScaler()

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model_tiny = TinyViTUperBUilder(cfg['net'], cfg["crop_size"])
    optimizer_tiny = SGD([
        {'params': model_tiny.encoder.parameters(), 'lr': cfg['lr']},
        {'params': [param for name, param in model_tiny.named_parameters() if 'encoder' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}], 
        lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('TinyViT Total params: {:.1f}M\n'.format(count_params(model_tiny)))
        
    local_rank = int(os.environ["LOCAL_RANK"])
    model_tiny = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_tiny)
    model_tiny.cuda()
    model_tiny = torch.nn.parallel.DistributedDataParallel(
        model_tiny, 
        device_ids=[local_rank], 
        broadcast_buffers=False,
        output_device=local_rank, 
        find_unused_parameters=True)


    model_cnn = ResUperBuilder(cfg['net'])
    optimizer_cnn = SGD([{'params': model_cnn.encoder.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model_cnn.named_parameters() if 'encoder' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    if rank == 0:
        logger.info('CNN Total params: {:.1f}M\n'.format(count_params(model_cnn)))
    model_cnn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_cnn)
    model_cnn.cuda()
    model_cnn = torch.nn.parallel.DistributedDataParallel(
        model_cnn, 
        device_ids=[local_rank], 
        broadcast_buffers=False,
        output_device=local_rank, 
        find_unused_parameters=False)
    

    model_vit = ViTUperBuilder(cfg['net'], cfg["crop_size"])
    optimizer_vit = SGD([{'params': model_vit.encoder.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model_vit.named_parameters() if 'encoder' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    if rank == 0:
        logger.info('ViT Total params: {:.1f}M\n'.format(count_params(model_vit)))
    model_vit = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_vit)
    model_vit.cuda()
    model_vit = torch.nn.parallel.DistributedDataParallel(
        model_vit, 
        device_ids=[local_rank], 
        broadcast_buffers=False,
        output_device=local_rank, 
        find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    # criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    criterion_u = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)

    trainset_u = SemiDataset_CPS(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset_CPS(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset_CPS(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'),map_location=torch.device('cpu'))
        model_tiny.load_state_dict(checkpoint['model_tiny'])
        optimizer_tiny.load_state_dict(checkpoint['optimizer_tiny'])
        model_cnn.load_state_dict(checkpoint['model_cnn'])
        optimizer_cnn.load_state_dict(checkpoint['optimizer_cnn'])
        model_vit.load_state_dict(checkpoint['model_vit'])
        optimizer_vit.load_state_dict(checkpoint['optimizer_vit'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer_tiny.param_groups[0]['lr'], previous_best))
        
        model_tiny.train()
        model_cnn.train()
        model_vit.train()

        total_loss = AverageMeter()
        sup_loss_tiny = AverageMeter()
        sup_loss_cnn = AverageMeter()
        sup_loss_vit = AverageMeter()
        unsup_loss_tiny = AverageMeter()
        unsup_loss_cnn = AverageMeter()
        unsup_loss_vit = AverageMeter()
        unsup_loss = AverageMeter()
        low_loss = AverageMeter()
        high_loss = AverageMeter()


        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        for i, ((img_l, mask_l), (img_u, mask_u)) in enumerate(loader):
            optimizer_tiny.zero_grad()
            optimizer_cnn.zero_grad()
            optimizer_vit.zero_grad()
            
            img_l, mask_l = img_l.cuda(), mask_l.cuda()
            img_u = img_u.cuda()
            # forward
            # 前向过程(model + loss)开启 autocast
            with autocast(enabled=True):
                pred_sup_tiny, feats_sup_tiny, attns_sup_tiny = model_tiny(img_l)
                pred_unsup_tiny, feats_unsup_tiny, attns_unsup_tiny = model_tiny(img_u)

                pred_sup_cnn, feats_sup_cnn = model_cnn(img_l)
                pred_unsup_cnn, feats_unsup_cnn = model_cnn(img_u)

                pred_sup_vit, attns_sup_vit = model_vit(img_l)
                pred_unsup_vit, attns_unsup_vit = model_vit(img_u)

                # supervised loss
                loss_sup_tiny = criterion_l(pred_sup_tiny, mask_l)
                loss_sup_cnn = criterion_l(pred_sup_cnn, mask_l)
                # loss_sup_vit = criterion_l(pred_sup_vit, mask_l)
                loss_sup_vit = criterion_u(pred_sup_vit, mask_l) # or using original CELOSS for vit
                if epoch == 0 and i == 0:
                    print("using original CELOSS for vit")

                ### cps loss ###
                _, max_tiny = torch.max(pred_unsup_tiny, dim=1)
                _, max_cnn = torch.max(pred_unsup_cnn, dim=1)
                _, max_vit = torch.max(pred_unsup_vit, dim=1)

                max_tiny = max_tiny.long()
                max_cnn = max_cnn.long()
                max_vit = max_vit.long()

                # cps_loss_tiny = criterion_u(pred_unsup_tiny, max_cnn) 
                # cps_loss_cnn = criterion_u(pred_unsup_cnn, max_tiny)
                # cps_loss = cps_loss_tiny + cps_loss_cnn 
                cps_loss_tiny = criterion_u(pred_unsup_tiny, max_cnn) + criterion_u(pred_unsup_tiny, max_vit)
                cps_loss_cnn = criterion_u(pred_unsup_cnn, max_vit) + criterion_u(pred_unsup_cnn, max_tiny)
                cps_loss_vit = criterion_u(pred_unsup_vit, max_cnn) + criterion_u(pred_unsup_vit, max_tiny)
                
                cps_loss = cps_loss_tiny + cps_loss_cnn + cps_loss_vit

                ### low-level loss ###
                feats_tiny = torch.cat([feats_sup_tiny, feats_unsup_tiny], dim=0)
                feats_cnn = torch.cat([feats_sup_cnn.detach(), feats_unsup_cnn.detach()], dim=0)

                low_mse_loss = nn.MSELoss()(feats_tiny, feats_cnn)

                ### high-level loss ###
                attn_h, attn_w = attns_sup_vit.size()[2:]
                # stage4
                attns_vit_stage4 = torch.cat([attns_sup_vit.detach(), attns_unsup_vit.detach()], dim=0).contiguous()
                attns_tiny_stage4 = torch.cat([attns_sup_tiny, attns_unsup_tiny], dim=0).contiguous()

                if attns_tiny_stage4.size()[2:] != (attn_h, attn_w):
                    attns_tiny_stage4 = F.interpolate(attns_tiny_stage4, (attn_h, attn_w), mode="bilinear", align_corners=True)
                p_s = F.log_softmax(attns_tiny_stage4, dim=-1)
                p_t = F.softmax(attns_vit_stage4, dim=-1)
                high_kl_loss = nn.KLDivLoss()(p_s, p_t)

                ### Total Loss ###
                cps_weight = cfg["cps_weight"]
                low_weight = cfg["criterion_low"]["weight"]
                high_weight = cfg["criterion_high"]["weight"]

                tiny_weight = cfg["net"]["encoder_tiny"]["weight"]
                cnn_weight = cfg["net"]["encoder_cnn"]["weight"]
                vit_weight = cfg["net"]["encoder_vit"]["weight"]

                if i==0 and rank==0:
                    print("tiny_weight: ", tiny_weight, "\t cnn_weight: ", cnn_weight, "\t vit_weight: ", vit_weight, "\t cps_weight: ", cps_weight, "\t low_weight: ", low_weight, "\t high_weight: ", high_weight)
                loss = loss_sup_tiny * tiny_weight + \
                    loss_sup_cnn * cnn_weight + \
                    loss_sup_vit * vit_weight + \
                    cps_loss * cps_weight + \
                    low_mse_loss * low_weight + \
                    high_kl_loss * high_weight
               
       
                torch.distributed.barrier()

            # optimizer_tiny.zero_grad()
            # optimizer_cnn.zero_grad()
            # loss.backward()
            # optimizer_tiny.step()
            # optimizer_cnn.step()

            # Scales loss. 为了梯度放大
            scaler.scale(loss).backward()
            scaler.step(optimizer_tiny)
            scaler.step(optimizer_cnn)
            scaler.step(optimizer_vit)
            scaler.update()

            total_loss.update(loss.item())
            sup_loss_tiny.update(loss_sup_tiny.item())
            sup_loss_cnn.update(loss_sup_cnn.item())
            sup_loss_vit.update(loss_sup_vit.item())
            unsup_loss_tiny.update(cps_loss_tiny.item())
            unsup_loss_cnn.update(cps_loss_cnn.item())
            unsup_loss_vit.update(cps_loss_vit.item())
            unsup_loss.update(cps_loss.item())
            low_loss.update(low_mse_loss.item())
            high_loss.update(high_kl_loss.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer_tiny.param_groups[0]["lr"] = lr
            optimizer_tiny.param_groups[1]["lr"] = lr * cfg['lr_multi']
            optimizer_cnn.param_groups[0]["lr"] = lr
            optimizer_cnn.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_sup_tiny', loss_sup_tiny.item(), iters)
                writer.add_scalar('train/loss_sup_cnn', loss_sup_cnn.item(), iters)
                writer.add_scalar('train/loss_sup_vit', loss_sup_vit.item(), iters)
                writer.add_scalar('train/cps_loss_tiny', cps_loss_tiny.item(), iters)
                writer.add_scalar('train/cps_loss_cnn', cps_loss_cnn.item(), iters)
                writer.add_scalar('train/cps_loss_vit', cps_loss_vit.item(), iters)
                writer.add_scalar('train/cps_loss', cps_loss.item(), iters)
                writer.add_scalar('train/low_mse_loss', low_mse_loss.item(), iters)
                writer.add_scalar('train/high_kl_loss', high_kl_loss.item(), iters)
               
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, sup_loss_tiny: {:.3f}, sup_loss_cnn: {:.3f}, sup_loss_vit: {:.3f}, unsup_loss_tiny: {:.3f}, unsup_loss_cnn: {:.3f}, unsup_loss_vit: {:.3f}, unsup_loss: {:.3f}, low_loss: {:.3f}, high_loss: {:.3f}'
                            .format(i, total_loss.avg, sup_loss_tiny.avg, sup_loss_cnn.avg, sup_loss_vit.avg,
                                            unsup_loss_tiny.avg, unsup_loss_cnn.avg, unsup_loss_vit.avg, unsup_loss.avg, low_loss.avg, high_loss.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        # 修改这里做模型消融
        # mIoU_tiny, iou_class_tiny = evaluate(model_tiny, valloader, eval_mode, cfg)
        mIoU_tiny, iou_class_tiny = evaluate(model_tiny, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class_tiny):
                logger.info('***** Evaluation TinyViT >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} TinyViT >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU_tiny))

            # for (cls_idx, iou) in enumerate(iou_class_cnn):
            #     logger.info('***** Evaluation CNN >>>> Class [{:} {:}] '
            #                 'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            # logger.info('***** Evaluation {} CNN >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU_cnn))
            
            writer.add_scalar('eval/mIoU_tiny', mIoU_tiny, epoch)
            # writer.add_scalar('eval/mIoU_cnn', mIoU_cnn, epoch)
            
            for i, iou in enumerate(iou_class_tiny):
                writer.add_scalar('eval_tiny/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
            
            # for i, iou in enumerate(iou_class_cnn):
            #     writer.add_scalar('eval_cnn/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best_tiny = mIoU_tiny > previous_best
        previous_best = max(mIoU_tiny, previous_best)
        if rank == 0:
            checkpoint = {
                'model_tiny': model_tiny.state_dict(),
                'optimizer_tiny': optimizer_tiny.state_dict(),
                'model_cnn': model_cnn.state_dict(),
                'optimizer_cnn': optimizer_cnn.state_dict(),
                'model_vit': model_vit.state_dict(),
                'optimizer_vit': optimizer_vit.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            checkpoint_tiny = {
                'model_tiny': model_tiny.state_dict(),
                'optimizer_tiny': optimizer_tiny.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint_tiny, os.path.join(args.save_path, 'tiny_last.pth'))
            if is_best_tiny:
                torch.save(checkpoint_tiny, os.path.join(args.save_path, 'tiny_best.pth'))


if __name__ == '__main__':
    main()
