"""
Inference module for insect detection on Yellow Sticky Traps
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from ultralytics import YOLO
import gc

def predict_on_image(model_path, image_path, conf_threshold=0.25):
    """
    Run inference on a single image
    
    Args:
        model_path: Path to the trained YOLO model
        image_path: Path to the image for prediction
        conf_threshold: Confidence threshold for predictions
        
    Returns:
        Prediction results
    """
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold)[0]
    return results

def predict_on_tiled_image(model_path, image_path, tile_size=1280, overlap=100, 
                          conf_threshold=0.25, iou_threshold=0.7, class_names=None):
    """
    Run inference on a large image by tiling and applying NMS on results
    
    Args:
        model_path: Path to trained YOLO model
        image_path: Path to the image
        tile_size: Size of tiles to process
        overlap: Overlap between adjacent tiles
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        class_names: List of class names for visualization
        
    Returns:
        Combined detection results and visualized image
    """
    model = YOLO(model_path)
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)

    stride = tile_size - overlap

    all_boxes = []
    all_scores = []
    all_classes = []

    # Process image by tiles
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            if x + tile_size > img_w:
                x = max(0, img_w - tile_size)
            if y + tile_size > img_h:
                y = max(0, img_h - tile_size)

            # Extract tile and run prediction
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            result = model(tile, conf=conf_threshold)[0]

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            # Adjust coordinates to original image
            for box, score, cls in zip(boxes, scores, classes):
                bx1, by1, bx2, by2 = box[:4]
                all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
                all_scores.append(score)
                all_classes.append(int(cls))
            
            # Clean up tile to free memory
            tile.close()

    # Convert to tensors for NMS
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores)
    classes_tensor = torch.tensor(all_classes)

    # Apply NMS per class
    keep_indices = []
    for class_id in torch.unique(classes_tensor):
        idxs = (classes_tensor == class_id).nonzero(as_tuple=True)[0]
        kept = torch.ops.torchvision.nms(boxes_tensor[idxs], scores_tensor[idxs], iou_threshold)
        keep_indices.extend(idxs[kept].tolist())

    # Draw results on the image
    for i in keep_indices:
        box = all_boxes[i]
        class_id = all_classes[i]
        score = all_scores[i]
        
        # Get label text
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]} {score:.2f}"
        else:
            label = f"Class {class_id} {score:.2f}"
            
        # Draw bounding box and label
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0] + 2, box[1] + 2), label, fill="white")

    # Clean up
    gc.collect()
    return image, keep_indices, [all_boxes[i] for i in keep_indices], [all_classes[i] for i in keep_indices], [all_scores[i] for i in keep_indices]

def visualize_results(image, title="Detection Results"):
    """
    Visualize detection results
    
    Args:
        image: PIL Image with detection boxes drawn
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def save_results(image, output_path):
    """
    Save detection results
    
    Args:
        image: PIL Image with detection boxes drawn
        output_path: Path to save the output image
    """
    image.save(output_path)
    print(f"Results saved to {output_path}")