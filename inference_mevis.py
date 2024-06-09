'''
Inference code for MUTR, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json

import opts
from tqdm import tqdm

import multiprocessing as mp
import threading

from tools.colormap import colormap
import warnings
warnings.filterwarnings('ignore')
import shutil

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
	T.Resize(432),
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


import zipfile
def zip_folder(source_folder, zip_dir):
	f = zipfile.ZipFile(zip_dir, 'w', zipfile.ZIP_DEFLATED)
	pre_len = len(os.path.dirname(source_folder))
	for dirpath, dirnames, filenames in os.walk(source_folder):
		for filename in filenames:
			pathfile = os.path.join(dirpath, filename)
			arcname = pathfile[pre_len:].strip(os.path.sep)
			f.write(pathfile, arcname)
	f.close()


def main(args):
	args.masks = True
	args.batch_size == 1

	os.makedirs(args.output_dir, exist_ok=True)
	inference_ckpt = args.resume[-4:-9]
	print(inference_ckpt)

	args.model_ckpt = inference_ckpt
	print("Inference only supports for batch size = 1") 

	# fix the seed for reproducibility
	seed = args.seed + utils.get_rank()
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	split = args.split
	# save path
	output_dir = args.output_dir
	#save_path_prefix = os.path.join(output_dir, split)
	TEST_DATASET = 'refytvos2021'
	CKPT = args.model_ckpt[:-4]
	TEST_DATASET_SPLIT = split
	exp_name = args.backbone
	eval_name = '{}_{}_{}_ckpt_{}'.format(TEST_DATASET,
											TEST_DATASET_SPLIT,
											exp_name, CKPT)
	# save_path_prefix = os.path.join(output_dir, '{}'.format(eval_name), 'Annotations')
	save_path_prefix = output_dir
	if not os.path.exists(save_path_prefix):
		os.makedirs(save_path_prefix)

	save_visualize_path_prefix = os.path.join(output_dir, '{}'.format(eval_name) + '_images')
	if args.visualize:
		if not os.path.exists(save_visualize_path_prefix):
			os.makedirs(save_visualize_path_prefix)

    # load data
	root = Path(args.mevis_path)        # mevis
	img_folder = os.path.join(root, split, "JPEGImages")
	meta_file = os.path.join(root, split, "meta_expressions.json")
	with open(meta_file, "r") as f:
		data = json.load(f)["videos"]
	valid_test_videos = set(data.keys())
	valid_videos = valid_test_videos    # - test_videos
	video_list = sorted([video for video in valid_videos])

	# create subprocess
	thread_num = args.ngpu
	global result_dict
	result_dict = mp.Manager().dict()

	processes = []
	lock = threading.Lock()

	video_num = len(video_list)
	per_thread_video_num = video_num // thread_num

	start_time = time.time()
	print('Start inference')
	for i in range(thread_num):
		if i == thread_num - 1:
			sub_video_list = video_list[i * per_thread_video_num:]
		else:
			sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
		p = mp.Process(target=sub_processor, args=(lock, i, args, data, 
												   save_path_prefix, 
												   img_folder, sub_video_list))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()

	end_time = time.time()
	total_time = end_time - start_time

	result_dict = dict(result_dict)
	num_all_frames_gpus = 0
	for pid, num_all_frames in result_dict.items():
		num_all_frames_gpus += num_all_frames

	print("Total inference time: %.4f s" %(total_time))

def sub_processor(lock, pid, args, data, save_path_prefix, img_folder, video_list):
	text = 'processor %d' % pid
	with lock:
		progress = tqdm(
			total=len(video_list),
			position=pid,
			desc=text,
			ncols=0
		)
	torch.cuda.set_device(pid)
	context_all = args.context_all
	clip_len = args.clip_len

	# model
	model, _, _ = build_model(args) 
	device = args.device
	model.to(device)

	model_without_ddp = model
	n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

	if pid == 0:
		print('number of params:', n_parameters)

	if args.resume:
		checkpoint = torch.load(args.resume, map_location='cpu')
		missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
		unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
		if len(missing_keys) > 0:
			print('Missing Keys: {}'.format(missing_keys))
		if len(unexpected_keys) > 0:
			print('Unexpected Keys: {}'.format(unexpected_keys))
	else:
		raise ValueError('Please specify the checkpoint for inference.')

	# start inference
	num_all_frames = 0 
	model.eval()

	# 1. For each video
	for video in video_list:
		metas = [] # list[dict], length is number of expressions
		expressions = data[video]["expressions"]   
		expression_list = list(expressions.keys()) 
		num_expressions = len(expression_list)

		# read all the anno meta
		for i in range(num_expressions):
			meta = {}
			meta["video"] = video
			meta["exp"] = expressions[expression_list[i]]["exp"]
			meta["exp_id"] = expression_list[i]
			meta["frames"] = data[video]["frames"]
			metas.append(meta)
		meta = metas

		# 2. For each expression
		for i in range(num_expressions):
			video_name = meta[i]["video"]
			exp = meta[i]["exp"]
			exp_id = meta[i]["exp_id"]
			frame_names = meta[i]["frames"]

			slices = []
			num_sub_samples = int(len(frame_names) / clip_len)
			if num_sub_samples == 0:
				num_sub_samples = 1

			if len(frame_names) % clip_len != 0 and len(frame_names) > clip_len:
				num_sub_samples += 1

			for ni in range(num_sub_samples):
				if context_all:
					slices.append(slice(ni*clip_len, min((ni+1)*clip_len, len(frame_names)), 1))
				else:
					slices.append(slice(ni, len(frame_names), num_sub_samples))
					# uncomment if using continuous mode
					# slices.append(slice(ni*clip_len, min((ni+1)*clip_len, len(frame_names)), 1))

			mti_hs_box_list = []
			pred_masks_list = []
			sub_frame_name_list = []

			for si, s in enumerate(slices):
				sub_frame_names = frame_names[s]
				sub_video_len = len(sub_frame_names)

				# store images
				imgs = []
				for frame_name in sub_frame_names:
					img_path = os.path.join(img_folder, video_name, frame_name + ".jpg")
					img = Image.open(img_path).convert('RGB')
					origin_w, origin_h = img.size
					imgs.append(transform(img)) # list[img]

				imgs = torch.stack(imgs, dim=0).to(args.device) # [video_len, 3, h, w]
				img_h, img_w = imgs.shape[-2:]
				size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
				target = {"size": size}

				with torch.no_grad():
					outputs, mti_hs_box = model([imgs], [exp], [target])
					mti_hs_box_list.append(mti_hs_box)
				
				pred_logits = outputs["pred_logits"][0]   
				pred_masks = outputs["pred_masks"][0]

				if context_all:
					pred_masks_list.append(pred_masks)
					sub_frame_name_list.append(sub_frame_names)
					continue

				# according to pred_logits, select the query index
				pred_scores = pred_logits.sigmoid() # [t, q, k]
				pred_scores = pred_scores.mean(0)   # [q, k]
				max_scores, _ = pred_scores.max(-1) # [q,]
				_, max_ind = max_scores.max(-1)     # [1,]
				max_inds = max_ind.repeat(sub_video_len)
				pred_masks = pred_masks[range(sub_video_len), max_inds, ...] # [t, h, w]
				pred_masks = pred_masks.unsqueeze(0)

				pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
				pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy() 
				all_pred_masks = pred_masks

				# save binary image
				save_path = os.path.join(save_path_prefix, video_name, exp_id)
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				for j, frame_name in enumerate(sub_frame_names):
					mask = all_pred_masks[j].astype(np.float32) 
					mask = Image.fromarray(mask * 255).convert('L')
					save_file = os.path.join(save_path, frame_name + ".png")
					mask.save(save_file)

			if context_all:
				# interactions between text and the whole video
				with torch.no_grad():
					pred_logits = model.inference_long_term(mti_hs_box_list, t=len(frame_names))
					pred_scores = pred_logits.sigmoid() # [t, q, k]
					pred_scores = pred_scores.mean(0)   # [q, k]
					max_scores, _ = pred_scores.max(-1) # [q,]
					_, max_ind = max_scores.max(-1)     # [1,]
				
				for (sub_frame_names, pred_masks) in zip(sub_frame_name_list, pred_masks_list):
					sub_video_len = len(sub_frame_names)
					max_inds = max_ind.repeat(sub_video_len)
					pred_masks = pred_masks[range(sub_video_len), max_inds, ...] # [t, h, w]
					pred_masks = pred_masks.unsqueeze(0)
					pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
					pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy() 
					all_pred_masks = pred_masks

					# save binary image
					save_path = os.path.join(save_path_prefix, video_name, exp_id)
					if not os.path.exists(save_path):
						os.makedirs(save_path)
					for j, frame_name in enumerate(sub_frame_names):
						mask = all_pred_masks[j].astype(np.float32) 
						mask = Image.fromarray(mask * 255).convert('L')
						save_file = os.path.join(save_path, frame_name + ".png")
						mask.save(save_file)
						pass

		with lock:
			progress.update(1)
	result_dict[str(pid)] = num_all_frames
	with lock:
		progress.close()

	# if pid == 0:
	# 	TEST_DATASET = 'refytvos2021'
	# 	CKPT = args.model_ckpt[:-4]
	# 	TEST_DATASET_SPLIT = args.split
	# 	exp_name = args.backbone
	# 	eval_name = '{}_{}_{}_ckpt_{}'.format(TEST_DATASET,
	# 											TEST_DATASET_SPLIT, 
	# 											exp_name, CKPT)

	# 	source_folder = os.path.join('results' , args.backbone, 'eval', args.dataset_file, eval_name, 'Annotations')
	# 	zip_dir = os.path.join('results', args.backbone, 'eval', args.dataset_file, '{}.zip'.format(eval_name))
	# 	zip_folder(source_folder, zip_dir)
	
	# 	print('Saving result to {}.'.format(zip_dir))


# visuaize functions
def box_cxcywh_to_xyxy(x):
	x_c, y_c, w, h = x.unbind(1)
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
		 (x_c + 0.5 * w), (y_c + 0.5 * h)]
	return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
	img_w, img_h = size
	b = box_cxcywh_to_xyxy(out_bbox)
	b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
	return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
	W, H = img_size
	for i, ref_point in enumerate(reference_points):
		init_x, init_y = ref_point
		x, y = W * init_x, H * init_y
		cur_color = color
		draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
		draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
	alpha = 255
	for i, samples in enumerate(sample_points):
		for sample in samples:
			x, y = sample
			cur_color = color_list[i % len(color_list)][::-1]
			cur_color += [alpha]
			draw.ellipse((x-2, y-2, x+2, y+2), 
							fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color):
	origin_img = np.asarray(img.convert('RGB')).copy()
	color = np.array(color)

	mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
	mask = mask > 0.5

	origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
	origin_img = Image.fromarray(origin_img)
	return origin_img

  

if __name__ == '__main__':
	parser = argparse.ArgumentParser('MUTR inference script', parents=[opts.get_args_parser()])
	args = parser.parse_args()
	main(args)
