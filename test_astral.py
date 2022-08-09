
from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path
from collections import OrderedDict

import open3d as o3d
reg_module = o3d.pipelines.registration

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import math

import clearml
import os

from tqdm import tqdm
from astral_dataset_reader import AstralDatasetReader
from mldatatools.dataset import Dataset, Lidar, LidarConfig

import cv2

import plotly.graph_objects as go
from yolov6.data.data_augment import letterbox
import logging


def check_img_size(img_size, s=32, floor=0) -> Union[int, List[int]]:
    if isinstance(img_size, tuple):
        img_size = list(img_size)

    def make_divisible( x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size, list) else [new_size] * 2


def precess_image(source_image: np.ndarray, img_size, stride, half) -> Tuple[np.ndarray, np.ndarray]:
    image = letterbox(source_image, img_size, stride=stride)[0]

    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, source_image


def main(dataset: AstralDatasetReader, weights_path: Union[Path, str] = 'checkpoints/LCDNet-kitti360.tar'):
    from yolov6.utils.events import LOGGER, load_yaml
    from yolov6.layers.common import DetectBackend
    from yolov6.utils.nms import non_max_suppression
    from yolov6.core.inferer import Inferer

    checkpoint: str = "yolov6s"  # @param ["yolov6s", "yolov6n", "yolov6t"]
    device: str = "gpu"  # @param ["gpu", "cpu"]
    half: bool = False  # @param {type:"boolean"}

    img_size: int = 640  # @param {type:"integer"}

    conf_thres: float = .25  # @param {type:"number"}
    iou_thres: float = .45  # @param {type:"number"}
    max_det: int = 1000  # @param {type:"integer"}
    agnostic_nms: bool = False  # @param {type:"boolean"}

    if not os.path.exists(f"{checkpoint}.pt"):
        print("Downloading checkpoint...")
        os.system(f"""wget -c https://github.com/meituan/YOLOv6/releases/download/0.1.0/{checkpoint}.pt""")
    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    model = DetectBackend(f"./{checkpoint}.pt", device=device)
    stride = model.stride
    class_names = load_yaml("./data/coco.yaml")['names']

    frame = dataset[0]
    image = frame['fc_far']
    img_size = check_img_size(image.shape[:2], s=stride)

    if half & (device.type != 'cpu'):
        model.model.half()
    else:
        model.model.float()
        half = False

    if device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup

    for frame_index in range(300, len(dataset)):
        frame = dataset[frame_index]
        image = frame['fc_far']
        img, img_src = precess_image(image, img_size, stride, half)
        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        pred_results = model(img)
        classes: Optional[List[int]] = None  # the classes to keep
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        img_ori = img_src.copy()
        if len(det):
            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls)
                label = f'{class_names[class_num]} {conf:.2f}'
                Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label,
                                           color=Inferer.generate_colors(class_num, True))
        img_ori = cv2.resize(img_ori, None, None, 0.5, 0.5)
        cv2.imshow('fc_far', img_ori)
        cv2.waitKey(13)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='', help="Path to dataset")
    parser.add_argument('--dataset_id', default='', help='Dataset id')
    parser.add_argument('--weights_path', default='checkpoints/LCDNet-kitti360.tar', help='Path to model weights')
    args = parser.parse_args()

    if args.dataset_id:
        clearml_dataset = clearml.Dataset.get(dataset_id=args.dataset_id)
        dataset_path = Path(clearml_dataset.get_local_copy())
    elif args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        raise 'Dataset is not set - "dataset_path" or "dataset_id" argument is needed'

    dataset = AstralDatasetReader(dataset_path, 'm11', ['fc_far'])

    main(dataset, args.weights_path)
