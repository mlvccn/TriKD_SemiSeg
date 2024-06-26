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
from model.semseg.model_helper_kd import TinyViTUperBUilder, ViTUperBuilder
from supervised_tiny import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from matplotlib.colors import Normalize
from ctypes import sizeof
import logging
import os
import time
from argparse import ArgumentParser

import numpy as np
from numpy.core.fromnumeric import size
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# from u2pl.models.model_helper_head import TinyViTUperBUilder, ResUperBuilder, ViTUperBuilder
from util.eval_helper import (
    AverageMeter,
    check_makedirs,
    colorize,
    convert_state_dict,
    create_cityscapes_label_colormap,
    create_pascal_label_colormap,
    intersectionAndUnion,
)


# Setup Parser
def get_parser():
    parser = ArgumentParser(description="PyTorch Evaluation")
    parser.add_argument(
        "--base_size", type=int, default=2048, help="based size for scaling"
    )
    parser.add_argument(
        "--scales", type=float, default=[1.0], nargs="+", help="evaluation scales"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/psp_best.pth",
        help="evaluation model path",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="checkpoints/results/",
        help="results save folder",
    )
    parser.add_argument(
        "--names_path",
        type=str,
        default="../../vis_meta/cityscapes/cityscapesnames.mat",
        help="path of dataset category names",
    )
    parser.add_argument(
        "--crop", action="store_true", default=False, help="whether use crop evaluation"
    )
    return parser


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger, cfg, colormap
    args = get_parser().parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = get_logger()
    logger.info(args)

    cfg_dset = cfg["dataset"]
    mean, std = cfg_dset["mean"], cfg_dset["std"]
    num_classes = cfg["net"]["num_classes"]
    crop_size = cfg_dset["val"]["crop"]["size"]
    crop_h, crop_w = crop_size

    assert num_classes > 1

    gray_folder = os.path.join(args.save_folder, "gray")
    color_folder = os.path.join(args.save_folder, "color")
    os.makedirs(gray_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)

    cfg_dset = cfg["dataset"]
    data_root, f_data_list = cfg_dset["val"]["data_root"], cfg_dset["val"]["data_list"]
    data_list = []

    if "Cityscapes" in data_root:
        colormap = create_cityscapes_label_colormap()
        for line in open(f_data_list, "r"):
            arr = [
                line.strip(),
                "gtFine/" + line.strip()[12:-15] + "gtFine_labelTrainIds.png",
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    else:
        colormap = create_pascal_label_colormap()
        for line in open(f_data_list, "r"):
            arr = [
                "JPEGImages/{}.jpg".format(line.strip()),
                "SegmentationClassAug/{}.png".format(line.strip()),
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    # Create network.
    args.use_auxloss = True if cfg["net"].get("aux_loss", False) else False
    logger.info("=> creating model from '{}' ...".format(args.model_path))

    cfg["net"]["sync_bn"] = False
    # Create network.
    img_size = cfg["dataset"]["train"]["crop"]["size"][0]
    print("img_size: ", img_size)

    model = TinyViTUperBUilder(cfg["net"], img_size)

    checkpoint = torch.load(args.model_path)
    for key in checkpoint:
        print(key)
    # saved_state_dict = convert_state_dict(checkpoint["model_state_tiny"]) # for semi-supervised 
    saved_state_dict = convert_state_dict(checkpoint["model_tiny"]) # for supervised
    # model.load_state_dict(saved_state_dict, strict=False)
    model.load_state_dict(saved_state_dict)
    model.cuda()
    logger.info("Load Model Done!")
    scales = [1.0,]
    if "cityscapes" in cfg["dataset"]["type"]:
        validate_city(
            model,
            num_classes,
            data_list,
            mean,
            std,
            args.base_size,
            crop_h,
            crop_w,
            # args.scales,
            scales,
            gray_folder,
            color_folder,
        )
    else:
        print("valiadte_whole")
        valiadte_whole(
            model,
            num_classes,
            data_list,
            mean,
            std,
            args.scales,
            gray_folder,
            color_folder,
            crop_size,
        )
    # cal_acc(data_list, gray_folder, num_classes)


@torch.no_grad()
def net_process(model, image):
    b, c, h, w = image.shape
    # num_classes = cfg['net']['num_classes']
    # output_all = torch.zeros((6, b, num_classes, h, w)).cuda()
    input = image.cuda()
    output ,feature = model(input)
    output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    # output_all[0] = F.softmax(output, dim=1)
    #
    # output = model(torch.flip(input, [3]))["pred"]
    # output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    # output = F.softmax(output, dim=1)
    # output_all[1] = torch.flip(output, [3])
    #
    # scales = [(961, 961), (841, 841), (721, 721), (641, 641)]
    # for k, scale in enumerate(scales):
    #     input_scale = F.interpolate(input, scale, mode="bilinear", align_corners=True)
    #     output = model(input_scale)["pred"]
    #     output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    #     output_all[k + 2] = F.softmax(output, dim=1)
    #
    # output = torch.mean(output_all, dim=0)
    return output,feature


def scale_crop_process(model, image, classes, crop_h, crop_w, h, w, stride_rate=2 / 3):
    ori_h, ori_w = image.size()[-2:]
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half)
        image = F.pad(image, border, mode="constant", value=0.0)
    new_h, new_w = image.size()[-2:]
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = torch.zeros((1, classes, new_h, new_w), dtype=torch.float).cuda()
    count_crop = torch.zeros((new_h, new_w), dtype=torch.float).cuda()
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[:, :, s_h:e_h, s_w:e_w].contiguous()
            count_crop[s_h:e_h, s_w:e_w] += 1
            with torch.no_grad():
                output,feature = net_process(
                    model, image_crop
                )
                prediction_crop[:, :, s_h:e_h, s_w:e_w] += output
    
    prediction_crop /= count_crop
    prediction_crop = prediction_crop[
        :, :, pad_h_half : pad_h_half + ori_h, pad_w_half : pad_w_half + ori_w
    ]
    prediction = F.interpolate(
        prediction_crop, size=(h, w), mode="bilinear", align_corners=True
    )
    return prediction[0] 


def scale_whole_process(model, image, h, w):
    with torch.no_grad():
        prediction = net_process(model, image)
    prediction = F.interpolate(
        prediction, size=(h, w), mode="bilinear", align_corners=True
    )
    return prediction[0]


def validate_city(
    model,
    classes,
    data_list,
    mean,
    std,
    base_size,
    crop_h,
    crop_w,
    scales,
    gray_folder,
    color_folder,
):
    global colormap
    logger.info(">>>>>>>>>>>>>>>> Start Crop Evaluation >>>>>>>>>>>>>>>>")
    data_time = AverageMeter()
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input_pth, label_path) in enumerate(data_list):
        data_time.update(time.time() - end)
        image = Image.open(input_pth).convert("RGB")
        image = np.asarray(image).astype(np.float32)
        label = Image.open(label_path).convert("L")
        label = np.asarray(label).astype(np.uint8)

        image = (image - mean) / std
        image = torch.Tensor(image).permute(2, 0, 1)
        image = image.contiguous().unsqueeze(dim=0)
        h, w = image.size()[-2:]
        # print("h: ", h, "; w: ", w)
        prediction = torch.zeros((classes, h, w), dtype=torch.float).cuda()
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)
            # print("new_h: ", new_h, "; new_w: ", new_w)
            image_scale = F.interpolate(
                image, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
            output = scale_crop_process(
                model, image_scale, classes, crop_h, crop_w, h, w
            )
            prediction += output
        image_path, _ = data_list[i]    
        image_name = image_path.split("/")[-1].split(".")[0]    
        prediction = torch.max(prediction, dim=0)[1].cpu().numpy()
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                    i + 1, len(data_list), data_time=data_time, batch_time=batch_time
                )
            )
        gray = np.uint8(prediction)
        color = colorize(gray, colormap)

        image_path, _ = data_list[i]
        image_name = image_path.split("/")[-1].split(".")[0]
        color_path = os.path.join(color_folder, image_name + ".png")
        color.save(color_path)
        intersection, union, target = intersectionAndUnion(gray, label, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, iou in enumerate(iou_class):
        logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
    logger.info(" * mIoU {:.2f}".format(np.mean(iou_class) * 100))
    logger.info("<<<<<<<<<<<<<<<<< End Crop Evaluation <<<<<<<<<<<<<<<<<")


def valiadte_whole(
    model, classes, data_list, mean, std, scales, gray_folder, color_folder, crop_size
):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input_pth, _) in enumerate(data_list):
        data_time.update(time.time() - end)
        image = Image.open(input_pth).convert("RGB")
        image = np.asarray(image).astype(np.float32)
        image = (image - mean) / std
        image = torch.Tensor(image).permute(2, 0, 1)
        # image = torch.Tensor(image)
        # print(image.size())
        ori_h, ori_w = image.size()[-2:]
        image = crop(crop_size, image)
        # print(image.size())
        image = image.contiguous().unsqueeze(dim=0)
        h, w = image.size()[-2:]
        prediction = torch.zeros((classes, h, w), dtype=torch.float).cuda()
        for scale in scales:
            new_h = round(h * scale)
            new_w = round(w * scale)
            image_scale = F.interpolate(
                image, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
            prediction += scale_whole_process(model, image_scale, h, w)
            print(f"prediction shape: ", prediction.shape)
        
        prediction = (
            torch.max(prediction, dim=0)[1].cpu().numpy()
        )  ##############attention###############
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                    i + 1, len(data_list), data_time=data_time, batch_time=batch_time
                )
            )
        check_makedirs(gray_folder)
        print(color_folder)
        check_makedirs(color_folder)
        print(f"ori_h: {ori_h}, ori_w: {ori_w}")
        gray = np.uint8(prediction)
        color = colorize(gray,colormap)
        # crop to return the original size
        print(f"color size: ", color.size)
        left = (w - ori_w) / 2 
        top = (h - ori_h)  / 2
        right = (w + ori_w) / 2
        bottom = (h + ori_h) / 2
        color = color.crop((left, top, right, bottom))
        print(f"color size: ", color.size)

        image_path, _ = data_list[i]
        image_name = image_path.split("/")[-1].split(".")[0]
        gray_path = os.path.join(gray_folder, image_name + ".png")
        color_path = os.path.join(color_folder, image_name + ".png")
        gray = Image.fromarray(gray)
        gray.save(gray_path)
        color.save(color_path)
    logger.info("<<<<<<<<<<<<<<<<< End  Evaluation <<<<<<<<<<<<<<<<<")


import collections
import random
def crop(size, image, crop_type="center"):
    if isinstance(size, int):
        crop_h = size
        crop_w = size
    elif (
        isinstance(size, collections.Iterable)
        and len(size) == 2
        and isinstance(size[0], int)
        and isinstance(size[1], int)
        and size[0] > 0
        and size[1] > 0
    ):
        crop_h = size[0]
        crop_w = size[1]
    else:
        raise (RuntimeError("crop size error.\n"))
    if crop_type == "center" or crop_type == "rand":
        crop_type = crop_type
    else:
        raise (RuntimeError("crop type error: rand | center\n"))
  
    
    h, w = image.size()[-2:]
    pad_h = max(crop_h - h, 0)
    pad_w = max(crop_w - w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half)
        image = F.pad(image, border, mode="constant", value=0.0)
   
    h, w = image.size()[-2:]
    if crop_type == "rand":
        h_off = random.randint(0, h - crop_h)
        w_off = random.randint(0, w - crop_w)
    else:
        h_off = (h - crop_h) // 2
        w_off = (w - crop_w) // 2
    image = image[:, h_off : h_off + crop_h, w_off : w_off + crop_w]
    return image



if __name__ == "__main__":
    main()