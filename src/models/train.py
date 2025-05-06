"""
Model training module for YOLO insect detection
"""
import os
from ultralytics import YOLO

def train_model(data_yaml_path, output_dir="runs/insect_detector", epochs=30, img_size=1280):
    """
    Train a YOLO model for insect detection
    
    Args:
        data_yaml_path: Path to YAML file with dataset configuration
        output_dir: Directory to save training results
        epochs: Number of training epochs
        img_size: Image size for training
    
    Returns:
        Path to the best trained model
    """
    # Initialize model
    model = YOLO("yolo11s.pt")  # Use YOLOv11 small as base model
    
    # Configure training parameters
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=0.8,  # Batch size based on available memory
        
        # Data augmentation parameters
        augment=True,
        mosaic=1.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.8,
        hsv_v=0.4,
        
        # Training scheduler parameters
        cos_lr=True,
        lr0=1e-3,
        lrf=0.01,
        optimizer='AdamW',
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        
        # Loss function parameters for imbalanced data
        cls=0.5,
        box=7.5,
        overlap_mask=True,
        single_cls=False,
        
        # Output directory configuration
        save_period=-1,
        project=os.path.dirname(output_dir),
        name=os.path.basename(output_dir),
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    
    # Return path to best model
    best_model_path = os.path.join(output_dir, "weights", "best.pt")
    return best_model_path

def evaluate_model(model_path, data_yaml_path):
    """
    Evaluate a trained YOLO model
    
    Args:
        model_path: Path to the trained model file
        data_yaml_path: Path to the data YAML file with validation data
        
    Returns:
        Evaluation metrics
    """
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml_path)
    print("✅ Evaluation complete")
    
    return metrics