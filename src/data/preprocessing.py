"""
Data processing module for tiling images and preparing datasets
"""
import os
import json
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import random
import gc

def convert_bbox_to_tile(xmin, ymin, xmax, ymax, tile_x, tile_y, tile_size, img_w, img_h):
    """Convert original bounding box coordinates to tile coordinates"""
    x_overlap = max(0, min(xmax, tile_x + tile_size) - max(xmin, tile_x))
    y_overlap = max(0, min(ymax, tile_y + tile_size) - max(ymin, tile_y))
    
    if x_overlap == 0 or y_overlap == 0:
        return None
        
    new_xmin = max(xmin - tile_x, 0)
    new_ymin = max(ymin - tile_y, 0)
    new_xmax = min(xmax - tile_x, tile_size)
    new_ymax = min(ymax - tile_y, tile_size)
    
    xc = (new_xmin + new_xmax) / 2 / tile_size
    yc = (new_ymin + new_ymax) / 2 / tile_size
    w = (new_xmax - new_xmin) / tile_size
    h = (new_ymax - new_ymin) / tile_size
    
    if w <= 0 or h <= 0:
        return None
        
    return xc, yc, w, h

def extract_tiles(img_path, xml_path, split, out_dir, class_map, tile_size, overlap):
    """Extract tiles from a single image and save them to the specific split folder"""
    tiles_created = 0
    filename = os.path.basename(img_path).replace('.jpg', '')
    
    # Parse XML outside of the image loading to reduce memory overlap
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    
    for obj in root.findall('object'):
        cls = obj.find('name').text.strip()
        if cls not in class_map:
            continue
            
        class_id = class_map[cls]
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        
        objects.append((class_id, xmin, ymin, xmax, ymax))
    
    # If no valid objects, don't even load the image
    if not objects:
        return 0
    
    # Now load the image since we have valid objects
    with Image.open(img_path) as img:
        img_w, img_h = img.size
        stride = tile_size - overlap

        for y in range(0, img_h, stride):
            if y + tile_size > img_h:
                y = max(0, img_h - tile_size)
                
            for x in range(0, img_w, stride):
                if x + tile_size > img_w:
                    x = max(0, img_w - tile_size)
                
                # Process tile only if it contains objects
                labels = []
                for class_id, xmin, ymin, xmax, ymax in objects:
                    bbox = convert_bbox_to_tile(xmin, ymin, xmax, ymax, x, y, tile_size, img_w, img_h)
                    if bbox:
                        labels.append(f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}")
                
                if not labels:
                    continue
                
                # Crop this specific tile
                tile = img.crop((x, y, x + tile_size, y + tile_size))
                tile_filename = f'{filename}_tile_{y}_{x}.jpg'
                label_filename = tile_filename.replace('.jpg', '.txt')
                
                # Save image
                tile.save(f"{out_dir}/images/{split}/{tile_filename}")
                
                # Save labels
                with open(f"{out_dir}/labels/{split}/{label_filename}", 'w') as f:
                    f.write('\n'.join(labels))
                
                # Update tile mapping data
                tile_map = {
                    'original_image': filename + '.jpg',
                    'tile_coord': [x, y, x + tile_size, y + tile_size],
                    'split': split
                }
                
                tiles_created += 1
                
                # Delete intermediate tile to free memory
                del tile
    
    return tiles_created

def process_dataset(src_img_dir, src_ann_dir, out_dir, classes, tile_size=1280, overlap=100, train_split=0.9, batch_size=5):
    """Process the entire dataset and create tiles"""
    # Create output directories
    os.makedirs(f'{out_dir}/images/train', exist_ok=True)
    os.makedirs(f'{out_dir}/images/val', exist_ok=True)
    os.makedirs(f'{out_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{out_dir}/labels/val', exist_ok=True)
    
    tile_map = {}
    class_map = {cls: i for i, cls in enumerate(classes)}
    
    # First split images into train/val, then process each set separately
    random.seed(42)  # For reproducible splits
    
    # Get all image files
    image_files = sorted(glob.glob(f'{src_img_dir}/*.jpg'))
    
    # Shuffle and split image files first
    random.shuffle(image_files)
    split_idx = int(train_split * len(image_files))
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Split {len(image_files)} images into {len(train_images)} training and {len(val_images)} validation images")
    
    # Process training images
    train_tiles = 0
    for i in range(0, len(train_images), batch_size):
        batch = train_images[i:i+batch_size]
        batch_tiles = 0
        
        for img_path in batch:
            xml_path = os.path.join(src_ann_dir, os.path.basename(img_path).replace('.jpg', '.xml'))
            if os.path.exists(xml_path):
                batch_tiles += extract_tiles(img_path, xml_path, 'train', out_dir, class_map, tile_size, overlap)
        
        train_tiles += batch_tiles
        print(f"Processed training batch {i//batch_size + 1}/{len(train_images)//batch_size + 1}: {batch_tiles} tiles created")
        gc.collect()  # Force garbage collection
    
    # Process validation images
    val_tiles = 0
    for i in range(0, len(val_images), batch_size):
        batch = val_images[i:i+batch_size]
        batch_tiles = 0
        
        for img_path in batch:
            xml_path = os.path.join(src_ann_dir, os.path.basename(img_path).replace('.jpg', '.xml'))
            if os.path.exists(xml_path):
                batch_tiles += extract_tiles(img_path, xml_path, 'val', out_dir, class_map, tile_size, overlap)
        
        val_tiles += batch_tiles
        print(f"Processed validation batch {i//batch_size + 1}/{len(val_images)//batch_size + 1}: {batch_tiles} tiles created")
        gc.collect()  # Force garbage collection
    
    # Save metadata
    with open(f'{out_dir}/tile_mapping.json', 'w') as f:
        json.dump(tile_map, f, indent=2)
    
    # Create dataset configuration file
    with open(f'{out_dir}/insects.yaml', 'w') as f:
        f.write(f"""train: {out_dir}/images/train
val: {out_dir}/images/val

nc: {len(classes)}
names: {classes}
""")
    
    # Final garbage collection
    gc.collect()
    
    print(f"✅ Done: {train_tiles} training and {val_tiles} validation tiles created at {out_dir}")
    return train_tiles, val_tiles