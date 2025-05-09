"""
Main script for insect detection on yellow sticky traps
This script provides a command-line interface for the insect detection pipeline
"""
import os
import argparse
import yaml
from data.preprocessing import process_dataset
from models.train import train_model, evaluate_model
from models.inference import predict_on_tiled_image, save_results
from utils.eda import analyze_dataset, plot_class_distribution, plot_bbox_dimensions

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Yellow Sticky Trap Insect Detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Dataset preparation command
    prep_parser = subparsers.add_parser("prepare", help="Prepare dataset by tiling images")
    prep_parser.add_argument("--img-dir", required=True, help="Directory with source images")
    prep_parser.add_argument("--ann-dir", required=True, help="Directory with source annotations")
    prep_parser.add_argument("--output-dir", required=True, help="Output directory for processed data")
    prep_parser.add_argument("--config", default="config/default.yaml", help="Configuration file")
    
    # EDA command
    eda_parser = subparsers.add_parser("eda", help="Run exploratory data analysis")
    eda_parser.add_argument("--img-dir", required=True, help="Directory with images")
    eda_parser.add_argument("--ann-dir", required=True, help="Directory with annotations")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train insect detection model")
    train_parser.add_argument("--data", required=True, help="Path to data YAML file")
    train_parser.add_argument("--output", default="runs/insect_detector", help="Output directory")
    train_parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    train_parser.add_argument("--img-size", type=int, default=1280, help="Image size for training")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    eval_parser.add_argument("--data", required=True, help="Path to data YAML file")
    
    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference on an image")
    infer_parser.add_argument("--model", required=True, help="Path to trained model")
    infer_parser.add_argument("--image", required=True, help="Path to input image")
    infer_parser.add_argument("--output", help="Path to save output image")
    infer_parser.add_argument("--tile-size", type=int, default=1280, help="Tile size for processing")
    infer_parser.add_argument("--overlap", type=int, default=100, help="Overlap between tiles")
    infer_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    infer_parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS")
    infer_parser.add_argument("--config", default="config/default.yaml", help="Configuration file with class names")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main function"""
    args = parse_args()
    
    if args.command == "prepare":
        config = load_config(args.config)
        process_dataset(
            src_img_dir=args.img_dir,
            src_ann_dir=args.ann_dir,
            out_dir=args.output_dir,
            classes=config['classes'],
            tile_size=config.get('tile_size', 1280),
            overlap=config.get('overlap', 100),
            train_split=config.get('train_split', 0.9),
            batch_size=config.get('batch_size', 5)
        )
    
    elif args.command == "eda":
        results = analyze_dataset(args.img_dir, args.ann_dir)
        print(f"Dataset Statistics:")
        print(f"Total Images: {results['total_images']}")
        print(f"Total Annotations: {results['total_annotations']}")
        print(f"Average Objects per Image: {sum(results['objects_per_image'])/len(results['objects_per_image']):.2f}")
        
        plot_class_distribution(results['label_counts'])
        plot_bbox_dimensions(results['bbox_dimensions'])
    
    elif args.command == "train":
        model_path = train_model(
            data_yaml_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            img_size=args.img_size
        )
        print(f"Training completed. Best model saved at: {model_path}")
    
    elif args.command == "evaluate":
        metrics = evaluate_model(args.model, args.data)
        print(f"Evaluation results: {metrics}")
    elif args.command == "infer":
        # Load class names from config
        config = load_config(args.config)
        class_names = config.get('classes', None)
        
        # Run inference
        result_image, keep_indices, boxes, classes, scores = predict_on_tiled_image(
            model_path=args.model,
            image_path=args.image,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            class_names=class_names
        )
        
        # Save output if specified
        if args.output:
            save_results(
                result_image, 
                args.output, 
                image_path=args.image, 
                boxes=boxes, 
                classes=classes,
                class_names=class_names
            )
        else:
            from models.inference import visualize_results
            visualize_results(result_image, f"Detection on {os.path.basename(args.image)}")
    
    else:
        print("Please specify a command. Use --help for more information.")

if __name__ == "__main__":
    main()