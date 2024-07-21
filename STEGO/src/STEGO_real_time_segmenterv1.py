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

# Paths and model loading
dir = "logs/checkpoints/stego_test_9/directory_carla_custom_1_clusterWarmup_L2_nclasses20_date_Jul15_14-27-25/"
sav_model = "epoch=24-step=14399.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model0 = LitUnsupervisedSegmenter.load_from_checkpoint(join(dir, sav_model)).to(device).eval()

vid_path = "../../testing_videos/"
video = "vid_town10HD_small_crash.mp4"
video_path = join(vid_path, video)

# Frame queue for real-time processing
frame_queue = queue.Queue(maxsize=10)  # Adjust maxsize based on your system capacity
segmented_frame_queue = queue.Queue(maxsize=10)

# Batch size for processing frames
BATCH_SIZE = 1  # Keep batch size 1 for real-time processing
resize_res = 448

# Function to process frames with STEGO
def process_batch_with_stego(frames, model, use_linear_probe=True):
    original_height, original_width = frames[0].shape[:2]
    transform = get_transform(resize_res, False, "center")  # Reduced resolution for processing
    img = transform(Image.fromarray(frames[0])).unsqueeze(0).cuda()

    with torch.no_grad():
        code1 = model(img)
        code2 = model(img.flip(dims=[3]))
        code  = (code1 + code2.flip(dims=[3])) / 2
        code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)
        linear_probs = None
        cluster_probs = None
        if use_linear_probe:
            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
        else:
            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

        segmented_frames = []
        single_img = img[0].cpu()
        # single_img = frame_tensors[i].cpu()
        # Apply CRF with linear_probs or cluster_probs as appropriate
        if use_linear_probe:  # Example condition, adjust as per your model's logic
            seg_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
        else:
            seg_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)
        
        # Convert segmentation prediction to colored image
        segmented_frame = model.label_cmap[seg_pred]
        # # segmented_frame = (segmented_frame * 255).astype(np.uint8)
        segmented_frame = segmented_frame.astype(np.uint8)
        # if segmented_frame.ndim == 2:
        #     segmented_frame = np.stack([segmented_frame] * 3, axis=-1)
        
        # # Resize back to original dimensions
        segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)
        segmented_frame = cv2.resize(segmented_frame, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        segmented_frames.append(segmented_frame)

    del img, code1, code2, code, linear_probs, cluster_probs, single_img, seg_pred
    torch.cuda.empty_cache()

    return segmented_frames


# Thread for reading video frames
def read_frames():
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count), desc="Reading Frames"):
        success, frame = cap.read()
        if not success:
            break
        frame_queue.put(frame)
    cap.release()
    frame_queue.put(None)  # Signal processing thread to exit

# Thread for processing frames
def process_frames():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        segmented_frames = process_batch_with_stego([frame], model0)
        for original_frame, segmented_frame in zip([frame], segmented_frames):
            segmented_frame_queue.put((original_frame, segmented_frame))

    segmented_frame_queue.put(None)  # Signal display thread to exit

# Thread for displaying combined frames
def display_segmented_frames():
    while True:
        frame_pair = segmented_frame_queue.get()
        if frame_pair is None:
            break
        original_frame, segmented_frame = frame_pair

        # Combine original and segmented frames side by side
        combined_frame = np.hstack((original_frame, segmented_frame))
        combined_frame = combined_frame.astype(np.uint8)

        cv2.imshow('Segmented Frame', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Create and start threads
reader_thread = threading.Thread(target=read_frames)
processor_thread = threading.Thread(target=process_frames)
display_thread = threading.Thread(target=display_segmented_frames)

reader_thread.start()
processor_thread.start()
display_thread.start()

reader_thread.join()
processor_thread.join()
display_thread.join()

print("Real-time video segmentation completed.")
