import argparse
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms


class RetinaDetector:
    def __init__(self, network, confidence=0.02, top_k=5000, nms_thresh=0.4, keep_top_k=750, vis_thresh=0.6):
        torch.set_grad_enabled(False)

        self.confidence = confidence
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.keep_top_k = keep_top_k
        self.vis_thresh = vis_thresh

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if network == "resnet":
            self.cfg = cfg_re50
            model_path = "weights/Resnet50_Final.pth"
        else:
            self.cfg = cfg_mnet
            model_path = "weights/mobilenet0.25_Final.pth"
        self.net = RetinaFace(cfg=self.cfg, phase='test')
        self.net = load_model(self.net, model_path,
                              True if self.device == "cpu" else False)
        self.net.eval()
        print('Finished loading model!')
        print(self.net)
        cudnn.benchmark = True
        self.device = torch.device(self.device)
        self.net = self.net.to(self.device)

    def detect(self, frame):
        resize = 1
        img = np.float32(frame)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(
            0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_thresh)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        results = []
        for det in dets:
            r = {}
            r["point"] = {}
            r["point"]["x1"] = int(det[0])
            r["point"]["y1"] = int(det[1])
            r["point"]["x2"] = int(det[2])
            r["point"]["y2"] = int(det[3])
            r["confidence"] = det[4]
            r["landmark"] = {}
            r["landmark"]["p1_x"] = int(det[5])
            r["landmark"]["p1_y"] = int(det[6])
            r["landmark"]["p2_x"] = int(det[7])
            r["landmark"]["p2_y"] = int(det[8])
            r["landmark"]["p3_x"] = int(det[9])
            r["landmark"]["p3_y"] = int(det[10])
            r["landmark"]["p4_x"] = int(det[11])
            r["landmark"]["p4_y"] = int(det[12])
            r["landmark"]["p5_x"] = int(det[13])
            r["landmark"]["p5_y"] = int(det[14])
            results.append(r)
        return results

    def write_bbox(self, frame, results, confidence=True, landmark=True):
        frame_copy = np.copy(frame)
        for r in results:
            if r["confidence"] < self.vis_thresh:
                continue
            text = "{:.4f}".format(r["confidence"])
            cv2.rectangle(frame_copy, (r["point"]["x1"], r["point"]["y1"]),
                            (r["point"]["x2"], r["point"]["y2"]), (0, 0, 255), 2)
            if confidence:
                cx = r["point"]["x1"]
                cy = r["point"]["y1"] + 12
                cv2.putText(frame_copy, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            if landmark:
                cv2.circle(frame_copy, (r["landmark"]["p1_x"], r["landmark"]["p1_y"]), 1, (0, 0, 255), 4)
                cv2.circle(frame_copy, (r["landmark"]["p2_x"], r["landmark"]["p2_y"]), 1, (0, 0, 255), 4)
                cv2.circle(frame_copy, (r["landmark"]["p3_x"], r["landmark"]["p3_y"]), 1, (0, 0, 255), 4)
                cv2.circle(frame_copy, (r["landmark"]["p4_x"], r["landmark"]["p4_y"]), 1, (0, 0, 255), 4)
                cv2.circle(frame_copy, (r["landmark"]["p5_x"], r["landmark"]["p5_y"]), 1, (0, 0, 255), 4)
        return frame_copy


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__=="__main__":
    retina = RetinaDetector("mobilenet")

    frame = cv2.imread("curve/test.jpg")

    results = retina.detect(frame)
    frame = retina.write_bbox(frame, results)
    cv2.imwrite("test.jpg", frame)

