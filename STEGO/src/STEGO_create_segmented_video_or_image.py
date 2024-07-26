import numpy as np
import torch
import torch.nn.functional as F
import cv2
import queue
import threading
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter
from utils import get_transform
from crf import dense_crf
from os.path import join
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
import os
from utils import unnorm, remove_axes

# Paths and model loading
dir = "logs/checkpoints/stego_test_9/directory_carla_custom_1_clusterWarmup_L2_nclasses20_date_Jul15_14-27-25/"
sav_model = "epoch=24-step=14399.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model0 = LitUnsupervisedSegmenter.load_from_checkpoint(join(dir, sav_model)).to(device).eval()

# Resolution video is 1024x512
vid_path = "../../testing_videos/"
video = "vid_T10HD.mp4"
video_path = join(vid_path, video)

segmented_video_path = "vid_segmented_T10HD.mp4"
segmented_video_save_path = join(vid_path, segmented_video_path)

segmented_groundtruth_video = "vid_segmentation_T10HD.mp4"
seg_gt_path = join(vid_path, segmented_groundtruth_video)

# Batch size for processing frames
BATCH_SIZE = 16
resize_res = 448

# Function to process frames with STEGO
def process_batch_with_stego(frames, model, use_linear_probe=True):
    original_height, original_width = frames[0].shape[:2]
    transform = get_transform(resize_res, False, "center")  # Reduced resolution for processing

    imgs = torch.stack([transform(Image.fromarray(frame)).cuda() for frame in frames])
    
    with torch.no_grad():
        code1 = model(imgs)
        code2 = model(imgs.flip(dims=[3]))
        code = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, imgs.shape[-2:], mode='bilinear', align_corners=False)
        linear_probs = None
        cluster_probs = None
        if use_linear_probe:
            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        else:
            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

        segmented_frames = []
        for i in range(len(frames)):
            single_img = imgs[i].cpu()
            if use_linear_probe:
                seg_pred = dense_crf(single_img, linear_probs[i]).argmax(0)
            else:
                seg_pred = dense_crf(single_img, cluster_probs[i]).argmax(0)
            
            segmented_frame = model.label_cmap[seg_pred]
            segmented_frame = segmented_frame.astype(np.uint8)
            segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)
            segmented_frame = cv2.resize(segmented_frame, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            segmented_frames.append(segmented_frame)

    del imgs, code1, code2, code, linear_probs, cluster_probs
    torch.cuda.empty_cache()

    return segmented_frames

# Function to read frames in a batch
def read_batch(cap, batch_size):
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

# Function to process and save video frames
def process_and_save_video():
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(segmented_video_save_path, fourcc, fps, (frame_width, frame_height))

    frame_queue = queue.Queue(maxsize=10)
    read_thread_running = True

    def read_thread():
        nonlocal read_thread_running
        while read_thread_running:
            if not frame_queue.full():
                frames = read_batch(cap, BATCH_SIZE)
                if not frames:
                    read_thread_running = False
                    break
                frame_queue.put(frames)
            else:
                cv2.waitKey(10)

    reader = threading.Thread(target=read_thread)
    reader.start()

    try:
        for _ in tqdm(range(0, frame_count, BATCH_SIZE), desc="Processing Video"):
            frames = frame_queue.get()
            segmented_frames = process_batch_with_stego(frames, model0)
            for seg_frame in segmented_frames:
                out.write(seg_frame)
            if not read_thread_running and frame_queue.empty():
                break
    finally:
        read_thread_running = False
        reader.join()

    cap.release()
    out.release()
    print(f"Segmented video saved at {segmented_video_save_path}")

# Function to calculate accuracy and Mean IoU
def calculate_metrics(segmented_video_path, ground_truth_video_path):
    cap_seg = cv2.VideoCapture(segmented_video_path)
    cap_gt = cv2.VideoCapture(ground_truth_video_path)

    frame_count_seg = int(cap_seg.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count_seg != frame_count_gt:
        print("The number of frames in the segmented video and ground truth video do not match.")
        return

    iou_scores = []
    accuracies = []

    for _ in tqdm(range(frame_count_seg), desc="Calculating Metrics"):
        success_seg, frame_seg = cap_seg.read()
        success_gt, frame_gt = cap_gt.read()
        if not success_seg or not success_gt:
            break

        # Convert frames to grayscale if necessary
        if frame_seg.ndim == 3:
            frame_seg = cv2.cvtColor(frame_seg, cv2.COLOR_BGR2GRAY)
        if frame_gt.ndim == 3:
            frame_gt = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2GRAY)

        # Calculate Intersection over Union (IoU)
        intersection = np.logical_and(frame_seg, frame_gt)
        union = np.logical_or(frame_seg, frame_gt)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)

        # Calculate accuracy
        correct = np.sum(frame_seg == frame_gt)
        total = frame_seg.size
        accuracy = correct / total
        accuracies.append(accuracy)

    cap_seg.release()
    cap_gt.release()

    mean_iou = np.mean(iou_scores)
    mean_accuracy = np.mean(accuracies)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Accuracy: {mean_accuracy:.4f}")

# Process and save the video
# process_and_save_video()

# Calculate metrics
# calculate_metrics(segmented_video_save_path, seg_gt_path)

def segment_and_show_image():
    model = model0

    j = 28
    i = 239
    # Load and transform the images
    img_path = '/storage/felix/Afstudeerproject/small_800-400_R07/imgs/train/train_'+ str(j)+ '_'+ str(i)+ '_bird_view_frame_RGB.jpg'
    img = Image.open(img_path)
    og_img = img
    original_height, original_width = np.array(img).shape[:2]

    transform = get_transform(448, False, "center")
    img = transform(img).unsqueeze(0).cuda()

    gt_img_path = '/storage/felix/Afstudeerproject/small_800-400_R07/labels/train/lab_train_'+ str(j)+ '_'+ str(i)+'_bird_view_frame_SEG.png'
    gt_img = Image.open(gt_img_path)
    og_gt = gt_img

    with torch.no_grad():
        code1 = model(img)
        code2 = model(img.flip(dims=[3]))
        code  = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
        linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

        single_img = img[0].cpu()
        linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
        cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)
        segmented_frame_lin = model.label_cmap[linear_pred].astype(np.uint8)
        segmented_frame_lin = cv2.resize(segmented_frame_lin, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        segmented_frame_clu = model.label_cmap[cluster_pred].astype(np.uint8)
        segmented_frame_clu = cv2.resize(segmented_frame_clu, (original_width, original_height), interpolation=cv2.INTER_NEAREST)


    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    # Display the images with correct aspect ratio
    ax[0].imshow(unnorm(img)[0].permute(1, 2, 0).cpu(), aspect='auto', extent=[0, img.shape[2], 0, img.shape[1]])
    ax[0].set_title("Image")
    ax[1].imshow(segmented_frame_clu, aspect='auto', extent=[0, cluster_pred.shape[1], 0, cluster_pred.shape[0]])
    ax[1].set_title("Cluster Predictions")
    ax[2].imshow(segmented_frame_lin, aspect='auto', extent=[0, linear_pred.shape[1], 0, linear_pred.shape[0]])
    ax[2].set_title("Linear Probe Predictions")

    # Remove the axes for a cleaner look
    remove_axes(ax)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


segment_and_show_image()
