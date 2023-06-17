import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from threading import Thread

FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # yolov5 strongsort root directory
print(sys.path)
ROOT = Path('/home/siddhu/turtlebot3_ws/src/yolo_tracker_dp_sort/yolo_tracker_dp_sort')
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
	sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
	sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker

#ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
bridge = CvBridge()

class Camera_subscriber(Node):

	def __init__(self):
		super().__init__('person_tracker')
		self.yolo_weights=WEIGHTS / 'yolov8s.pt'# model.pt path(s),
		self.reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt'  # model.pt path,
		self.tracking_method='deepocsort'
		self.tracking_config=ROOT/'trackers/deepocsort/configs/deepocsort.yaml'
		self.imgsz=(640, 640)  # inference size (height, width)
		self.conf_thres=0.25  # confidence threshold
		self.iou_thres=0.45  # NMS IOU threshold
		self.max_det=1000  # maximum detections per image

		self.classes=[0]  # filter by class: --class 0, or --class 0 2 3
		self.agnostic_nms=False  # class-agnostic NMS
		self.augment=False # augmented inference
		self.name='exp' # save results to project/name
		self.exist_ok=False  # existing project/name ok, do not increment
		self.line_thickness=2  # bounding box thickness (pixels)
		self.visualize = False
		self.half=False  # use FP16 half-precision inference
		self.dnn=False # use OpenCV DNN for ONNX inference
		if torch.cuda.is_available():

			# Initialize
			self.device = select_device('cuda:0')
		else:
			self.device = select_device('cpu')
		# Load model
		self.is_seg = '-seg' in str(self.yolo_weights)
		self.model = AutoBackend(self.yolo_weights, device=self.device, dnn=self.dnn, fp16=self.half)
		stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
		self.imgsz = check_imgsz(self.imgsz, stride=stride)  # check image size
		
		# Run inference
		bs = 1  # batch_size
		self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *self.imgsz))  # warmup
		
		# PUB SUB
		self.subscription = self.create_subscription(
			Image,
			'/webcam',
			self.camera_callback,
			5)
		self.subscription  # prevent unused variable warning
		self.publisher = self.create_publisher(Int32MultiArray, "/detections", 10)

		# Store Detections
		self.camera_fov = 60 #in degrees
		self.camera_image_size = (640,480)
		self.num_frames_undetected = 0
		
		# Create as many strong sort instances as there are video sources
		self.tracker_list = []
		for i in range(bs):
			tracker = create_tracker(self.tracking_method, self.tracking_config, self.reid_weights, self.device, self.half)
			self.tracker_list.append(tracker, )
			if hasattr(self.tracker_list[i], 'model'):
				if hasattr(self.tracker_list[i].model, 'warmup'):
					self.tracker_list[i].model.warmup()
		self.outputs = [None] * bs

		self.seen, windows, self.dt = 0, [], (Profile(), Profile(), Profile(), Profile())
		self.curr_frames, self.prev_frames = [None] * bs, [None] * bs
	@torch.no_grad()
	def camera_callback(self, data):
		'''Tracker Publishes Detection Bbox x1,x2,img_width'''
		img = bridge.imgmsg_to_cv2(data, "bgr8")
	
		# Display the resulting frame
		# cv2.imshow('frame', frame)
		
		# Letterbox
		img0 = img.copy()
		img = img[np.newaxis, :, :, :]        

		# Stack
		img = np.stack(img, 0)

		# Convert
		img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
		img = np.ascontiguousarray(img)

		img = torch.from_numpy(img).to(self.model.device)
		img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
		img /= 255  # 0 - 255 to 0.0 - 1.0
		if len(img.shape) == 3:
			img = img[None]  # expand for batch dim

		path = '' 
		im = img
		im0 = img0

		s = ''

		visualize = increment_path(self.save_dir / Path(path[0]).stem, mkdir=True) if self.visualize else False

		# Inference
		with self.dt[1]:
			preds = self.model(im, augment=self.augment, visualize=visualize)

		# Apply NMS
		with self.dt[2]:
			p = non_max_suppression(preds, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
			
		# Process detections
		for i, det in enumerate(p):  # detections per image
			self.seen += 1

			self.curr_frames[i] = im0

			# annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
			
			if hasattr(self.tracker_list[i], 'tracker') and hasattr(self.tracker_list[i].tracker, 'camera_update'):
				if self.prev_frames[i] is not None and self.curr_frames[i] is not None:  # camera motion compensation
					self.tracker_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])

			if det is not None and len(det):
				self.num_frames_undetected = 0
				det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

				# Print results
				for c in det[:, 5].unique():
					n = (det[:, 5] == c).sum()  # detections per class
					s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

				# pass detections to strongsort
				with self.dt[3]:
					self.outputs[i] = self.tracker_list[i].update(det.cpu(), im0)
				
				# draw boxes for visualization
				if len(self.outputs[i]) > 0:
					
					# create a list of all ids take the lowest one and return the B-box
					min_idx = 0
					min_id = self.outputs[i][min_idx][4]
					for j, (output) in enumerate(self.outputs[i]):
						if output[4] < min_id:
							min_id = output[4]
							min_idx = j
					

					# for j, (output) in enumerate(outputs[i]):
					j = min_idx
					output = self.outputs[i][j]	
					bbox = output[0:4]
					id = output[4]
					cls = output[5]
					conf = output[6]
					bbox = [int(i) for i in bbox]
					# image = cv2.rectangle(im0.copy(), (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0,0), 2)
					# image = cv2.putText(image, f'ID: {id} Class: {cls}', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
					# cv2.imshow('output',image)
					# cv2.waitKey(10)
					# thread = Thread(target=view,args=(im0,bbox,id,cls))
					# thread.start()
					h,w = im0.shape[0],im0.shape[1]
					bbox+=[h,w]
					print(f'Found a Person \nid:{id} \n\t\tLeft\t\tTop\tRight\t\tBelow \nB-Box: {output[0:4]} \nImg-size: {im0.shape}')
					view(im0,bbox,id,cls)
					x1,x2 = bbox[0],bbox[2]
					# Publish
					int_array = Int32MultiArray(data=[x1,x2,w])
					self.publisher.publish(int_array)
			else:
				# No detections
				self.num_frames_undetected += 1
				if self.num_frames_undetected>10:
					print(f'We have lost detection, since {self.num_frames_undetected} frames')
					int_array = Int32MultiArray(data=[])
					self.publisher.publish(int_array)
				#tracker_list[i].tracker.pred_n_update_all_tracks()


def view(img,bbox,id,cls):
	image = cv2.rectangle(img.copy(), (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255,0,0), 2)
	image = cv2.putText(image, f'ID: {id} Class: {cls}', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
	cv2.imshow('output',image)
	cv2.waitKey(1)
	
# check if the number of frames undetected is greater than 10 it means we don't have any direction to go right now.
def main():
	rclpy.init(args=None)
	camera_subscriber = Camera_subscriber()
	rclpy.spin(camera_subscriber)
	rclpy.shutdown()
if __name__ == '__main__':
	main()
