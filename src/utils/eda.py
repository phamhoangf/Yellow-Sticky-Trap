"""
Utilities for exploratory data analysis of insect images and annotations
"""
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
import pandas as pd
import numpy as np

def show_annotated_image(img_path, xml_path):
    """Display an image with its annotations overlaid"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    print("Image Height:", h)
    print("Image Width:", w)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(img, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def analyze_dataset(img_dir, ann_dir):
    """Analyze the dataset and return statistics"""
    # Initialize statistics variables
    label_counts = defaultdict(int)
    object_per_image = []
    box_dims = []

    # Process all annotation files
    ann_files = sorted(os.listdir(ann_dir))
    img_files = sorted(os.listdir(img_dir))

    for ann_file in ann_files:
        tree = ET.parse(os.path.join(ann_dir, ann_file))
        root = tree.getroot()
        
        objects = root.findall("object")
        object_per_image.append(len(objects))

        for obj in objects:
            label = obj.find("name").text
            label_counts[label] += 1

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            width = xmax - xmin
            height = ymax - ymin
            box_dims.append((width, height))

    # Convert to DataFrame for analysis
    box_df = pd.DataFrame(box_dims, columns=["width", "height"])
    box_df["area"] = box_df["width"] * box_df["height"]
    box_df["aspect_ratio"] = box_df["width"] / box_df["height"]

    # Return analysis results
    results = {
        "total_images": len(img_files),
        "total_annotations": len(ann_files),
        "label_counts": label_counts,
        "objects_per_image": object_per_image,
        "bbox_dimensions": box_df
    }
    
    return results

def plot_class_distribution(label_counts):
    """Plot the distribution of insect classes"""
    # Convert label counts to sorted lists
    labels = list(label_counts.keys())
    counts = [label_counts[label] for label in labels]

    # Sort by count in descending order
    sorted_data = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_data]
    counts = [item[1] for item in sorted_data]

    # Calculate percentages
    total = sum(counts)
    percentages = [(count/total)*100 for count in counts]

    # Create a color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=None, autopct='%1.1f%%', startangle=90, 
            colors=colors, shadow=False, explode=[0.05]*len(labels))

    plt.title('Insect Type Distribution', fontsize=16)

    # Add legend with class name and count
    legend_labels = [f'{label} ({count}, {percentage:.1f}%)' for label, count, percentage in zip(labels, counts, percentages)]
    plt.legend(legend_labels, loc='best', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Print summary information
    print(f"Total objects: {total}")
    print(f"Number of classes: {len(labels)}")
    print("\nDetailed distribution:")
    for label, count, percentage in zip(labels, counts, percentages):
        print(f"{label}: {count} ({percentage:.1f}%)")

def plot_bbox_dimensions(box_df):
    """Plot the distribution of bounding box dimensions"""
    # Calculate statistics for width
    widths = box_df["width"]
    w_min = widths.min()
    w_max = widths.max()
    w_counts, w_bins = np.histogram(widths, bins=30)
    w_peak_idx = np.argmax(w_counts)
    w_peak_value = (w_bins[w_peak_idx] + w_bins[w_peak_idx + 1]) / 2

    # Calculate statistics for height
    heights = box_df["height"]
    h_min = heights.min()
    h_max = heights.max()
    h_counts, h_bins = np.histogram(heights, bins=30)
    h_peak_idx = np.argmax(h_counts)
    h_peak_value = (h_bins[h_peak_idx] + h_bins[h_peak_idx + 1]) / 2

    # Create plots
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30, color='salmon')
    plt.title(f"Bounding Box Width\nMin: {w_min:.2f}, Max: {w_max:.2f}, Peak: {w_peak_value:.2f}")

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30, color='seagreen')
    plt.title(f"Bounding Box Height\nMin: {h_min:.2f}, Max: {h_max:.2f}, Peak: {h_peak_value:.2f}")

    plt.tight_layout()
    plt.show()