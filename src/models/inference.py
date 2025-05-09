"""
Inference module for insect detection on Yellow Sticky Traps
"""
import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from ultralytics import YOLO
import gc
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

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
                          conf_threshold=0.25, iou_threshold=0.3, class_names=None):
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
            tile.close()    # Convert to tensors for NMS
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores)
    classes_tensor = torch.tensor(all_classes)
    
    # Complete reimplementation using distance-based clustering
    final_keep_indices = []
    
    # Process each class separately
    for class_id in torch.unique(classes_tensor):
        # First get all boxes for this class
        class_indices = (classes_tensor == class_id).nonzero(as_tuple=True)[0]
        
        if len(class_indices) == 0:
            continue
            
        # Store box centers and their corresponding indices
        centers = []
        sizes = []
        indices = []
        
        # Calculate box centers and relative sizes
        for idx in class_indices:
            box = boxes_tensor[idx]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]
            centers.append((center_x, center_y))
            sizes.append((width, height))
            indices.append(idx.item())
        
        # Group boxes that are close together
        groups = []
        used = set()
        
        # Very aggressive distance threshold - based on average box size
        avg_width = sum(size[0] for size in sizes) / len(sizes)
        avg_height = sum(size[1] for size in sizes) / len(sizes)
        distance_threshold = min(avg_width, avg_height) * 0.6  # More aggressive
        
        # Form initial groups
        for i in range(len(indices)):
            if i in used:
                continue
                
            group = [i]
            used.add(i)
            
            for j in range(len(indices)):
                if j in used or i == j:
                    continue
                    
                # Calculate distance between centers
                dist = torch.sqrt((centers[i][0] - centers[j][0])**2 + 
                                 (centers[i][1] - centers[j][1])**2)
                
                # If close enough, add to group
                if dist < distance_threshold:
                    group.append(j)
                    used.add(j)
                    
                    # Additionally check if boxes overlap significantly
                    box1 = boxes_tensor[indices[i]]
                    box2 = boxes_tensor[indices[j]]
                    
                    # Calculate IoU
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])
                    
                    if x2 > x1 and y2 > y1:
                        # Calculate overlap
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                        
                        # If there's significant overlap, look for other boxes
                        # that might be close to this one too
                        if intersection / min(area1, area2) > 0.1:  # Very low IoU threshold
                            for k in range(len(indices)):
                                if k in used or k == i or k == j:
                                    continue
                                    
                                box3 = boxes_tensor[indices[k]]
                                
                                # Calculate distances to the overlapping boxes
                                dist_to_j = torch.sqrt((centers[j][0] - centers[k][0])**2 + 
                                                      (centers[j][1] - centers[k][1])**2)
                                
                                if dist_to_j < distance_threshold * 1.5:  # Even more aggressive
                                    group.append(k)
                                    used.add(k)
            
            groups.append(group)
        
        # For each group, keep only the box with highest confidence
        for group in groups:
            if not group:  # Skip empty groups
                continue
                
            # Find box with highest confidence in the group
            best_idx = max(group, key=lambda idx: scores_tensor[indices[idx]].item())
            final_keep_indices.append(indices[best_idx])
    
    # Final result is only the highest confidence box from each group
    keep_indices = final_keep_indices    # Draw results on the image
    try:
        # Try to load a larger font
        font = ImageFont.truetype("arial.ttf", 20)  # Larger font size
    except IOError:
        # Fallback if font not available
        font = ImageFont.load_default()
        
    for i in keep_indices:
        box = all_boxes[i]
        class_id = all_classes[i]
        score = all_scores[i]
        
        # Get label text
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]} {score:.2f}"
        else:
            label = f"Class {class_id} {score:.2f}"
            
        # Draw bounding box
        draw.rectangle(box, outline="red", width=3)
        
        # Calculate text position (above the box)
        text_position = (box[0], box[1] - 30)  # 30 pixels above the box
          # Add a background rectangle for text for better visibility
        # Use the modern approach to get text size with ImageFont.getbbox
        bbox = font.getbbox(label)  # Returns (left, top, right, bottom)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.rectangle([text_position[0], text_position[1], 
                       text_position[0] + text_width + 10, text_position[1] + text_height + 10], 
                       fill="black")
        
        # Draw label text in red with larger font
        draw.text(text_position, label, fill="red", font=font)

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
    
def save_results(image, output_path, image_path=None, boxes=None, classes=None, class_names=None):
    """
    Save detection results (image and optional annotation)
    
    Args:
        image: PIL Image with detection boxes drawn
        output_path: Path to save the output image
        image_path: Original image path (for annotation)
        boxes: List of bounding boxes (for annotation)
        classes: List of class indices (for annotation)
        class_names: List of class names
    """
    # Save result image
    image.save(output_path)
    print(f"Results saved to {output_path}")
    
    # If we have bounding box data and original image path, save XML annotation
    if image_path and boxes is not None and classes is not None:
        # Generate annotation file path by replacing image extension with .xml
        annotation_path = os.path.splitext(output_path)[0] + '.xml'
        save_annotation_as_xml(image_path, annotation_path, boxes, classes, class_names)

def save_annotation_as_xml(image_path, output_path, boxes, classes, class_names=None):
    """
    Save detection results as an XML annotation file in Pascal VOC format
    
    Args:
        image_path: Path to the original image
        output_path: Path to save the XML annotation file
        boxes: List of bounding boxes [x1, y1, x2, y2]
        classes: List of class indices
        class_names: List of class names
    """
    # Get image properties
    img = Image.open(image_path)
    width, height = img.size
    
    # Create XML annotation structure
    root = ET.Element("annotation")
    
    # Add basic image information
    folder = ET.SubElement(root, "folder")
    folder.text = os.path.dirname(image_path).split(os.path.sep)[-1]
    
    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)
    
    path = ET.SubElement(root, "path")
    path.text = image_path
    
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "dataset"
    
    # Add image size information
    size = ET.SubElement(root, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    
    # Add segmented flag
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    
    # Add objects (detections)
    for i in range(len(boxes)):
        box = boxes[i]
        class_id = classes[i]
        
        # Get class name
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class_{class_id}"
        
        # Create object element
        obj = ET.SubElement(root, "object")
        
        # Pascal VOC format uses 'name' field, but adapting to the sample format which uses 'n'
        name = ET.SubElement(obj, "n")
        name.text = class_name
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        # Add bounding box coordinates
        bndbox = ET.SubElement(obj, "bndbox")
        
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(box[0]))
        
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(box[1]))
        
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(box[2]))
        
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(box[3]))
    
    # Convert to string with proper formatting
    xml_str = ET.tostring(root, encoding="utf-8")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="\t")
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)
    
    print(f"Annotation saved to {output_path}")