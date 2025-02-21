import argparse
import os
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import cv2
from PIL import Image
import numpy as np
import tqdm  # For progress bar

# 1. Load Pre-trained Model and Processor
model_name = "facebook/detr-resnet-50"  # Or try "facebook/deformable-detr-resnet50"
processor = DetrImageProcessor.from_pretrained(model_name)  # For image preprocessing
model = DetrForObjectDetection.from_pretrained(model_name) # The ViT object detection model

# 2. Move Model to GPU (Very Important for your 3090 Ti rig!)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")  # Verify GPU is being used

# 3. Parse Command-Line Arguments
parser = argparse.ArgumentParser(description="Process a video for object detection.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
args = parser.parse_args()

# 4. Load your Video
video_path = args.video_path
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video frame rate (for later saving if needed)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output directory if it doesn't exist
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# Output video writer (optional - to save processed video)
output_video_path = os.path.join(output_dir, "detected_video.mp4") # Change extension if needed, .mp4 is generally good
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or try 'XVID'
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Object category labels (from the pre-trained model) - DETR uses COCO categories
id2label = model.config.id2label # Dictionary mapping IDs to category names
# print(id2label) # Uncomment to see the categories

# 5. Process Video Frame by Frame
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
progress_bar = tqdm.tqdm(total=total_frames, desc="Processing Frames")

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret: # End of video
        break

    frame_count += 1

    # 6. Preprocess Frame for ViT
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # Convert BGR (OpenCV) to RGB (PIL/ViT)
    inputs = processor(images=image, return_tensors="pt").to(device) # Preprocess and move to GPU

    # 7. Perform Object Detection (Inference)
    with torch.no_grad(): # Disable gradient calculations during inference for speed
        outputs = model(**inputs) # **inputs unpacks the input dictionary

    # 8. Post-process Detections
    target_sizes = torch.tensor([image.size[::-1]]).to(device) # (width, height) to (height, width)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0] # Threshold for confidence

    # 9. Draw Bounding Boxes and Labels on Frame
    for score, label_id, bbox in zip(results["scores"], results["labels"], results["boxes"]):
        score = score.item()
        label = id2label[label_id.item()]
        bbox = [int(i) for i in bbox.tolist()] # Convert bbox to integers (pixels)

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2) # Green box
        # Draw label and confidence score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(frame, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 10. Display or Save the Processed Frame (Optional Display, Recommended Save for Video)
    # cv2.imshow('Object Detection', frame) # Can display, but slows down processing significantly
    out_video.write(frame) # Write frame to output video

    progress_bar.update(1) # Update progress bar
    # if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit (if using cv2.imshow)
    #     break

progress_bar.close()
cap.release()
out_video.release()
cv2.destroyAllWindows()
print(f"Processed video saved to: {output_video_path}")