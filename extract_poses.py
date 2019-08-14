import os,sys
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
from opt import opt

from dataloader import WebcamLoader, ImageLoader, DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import write_json

args = opt
args.dataset = 'coco'

from glob import glob

assert args.inputpath != ""
assert args.outputpath != ""
input_imgs = glob(os.path.join(args.inputpath, "**", "*.jpg"), recursive=True)
print('input imgs', len(input_imgs))
print(args.outputpath)

# Load YOLO model
print('Loading YOLO model..')
sys.stdout.flush()
det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
det_model.load_weights('models/yolo/yolov3-spp.weights')
det_model.net_info['height'] = args.inp_dim
det_inp_dim = int(det_model.net_info['height'])
assert det_inp_dim % 32 == 0
assert det_inp_dim > 32
det_model.cuda()
det_model.eval()

print('load pose model')
# Load pose model
pose_dataset = Mscoco()
if args.fast_inference:
    pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
else:
    pose_model = InferenNet(4 * 1 + 1, pose_dataset)
pose_model.cuda()
pose_model.eval()

print('done')

# assume batch size 1 for now
data_loader = ImageLoader(input_imgs, batchSize=1, format='yolo').start()

data_len = data_loader.length()
im_names_desc = tqdm(range(data_len))

opt.confidence = 0.95

for i in im_names_desc:
    start_time = getTime()
    with torch.no_grad():
        (img, orig_img, im_name, im_dim_list) = data_loader.read()

        rel_img_path = os.path.relpath(im_name[0], args.inputpath)
        out_img_path = os.path.join(args.outputpath, rel_img_path)
        out_img_folder = os.path.dirname(out_img_path)
        os.makedirs(out_img_folder, exist_ok=True)

        # Human Detection
        img = Variable(img).cuda()
        im_dim_list = im_dim_list.cuda()

        prediction = det_model(img, CUDA=True)

        dets = dynamic_write_results(prediction, opt.confidence,
                     opt.num_classes, nms=True, nms_conf=opt.nms_thesh)

        if isinstance(dets, int) or dets.shape[0] == 0:
            # no human
            continue

        im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
        scaling_factor = torch.min(det_inp_dim / im_dim_list, 1)[0].view(-1, 1)
        # coordinate transfer
        dets[:, [1, 3]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        dets[:, [2, 4]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

        dets[:, 1:5] /= scaling_factor
        for j in range(dets.shape[0]):
            dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
            dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
        boxes = dets[:, 1:5].cpu()
        scores = dets[:, 5:6].cpu()

        inp = im_to_torch(cv2.cvtColor(orig_img[0], cv2.COLOR_BGR2RGB))
        # Pose Estimation
        inps = torch.zeros(boxes.size(0), 3, opt.inputResH, opt.inputResW)
        pt1 = torch.zeros(boxes.size(0), 2)
        pt2 = torch.zeros(boxes.size(0), 2)
        inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)

        #inps = Variable(inps.cuda())
        #hm = pose_model(inps)

        imgs = inps.cpu().numpy()
        for i in range(len(inps)):
            img = imgs[i]
            img[0] += 0.406
            img[1] += 0.457
            img[2] += 0.480
            img = (img * 255).astype(np.uint8)     # unnormalize
            img = np.transpose(img, (1, 2, 0))

            cv2.imwrite(out_img_path.replace(".jpg", "_{}.jpg".format(i)), img[..., ::-1])
