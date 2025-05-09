"""
Streamlit app for insect detection on Yellow Sticky Traps
This app provides a user interface for uploading images, running detection, and downloading results
"""
import os
import sys
import streamlit as st
from PIL import Image
import io
import tempfile
import uuid
import base64
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.inference import predict_on_tiled_image, save_annotation_as_xml

# Set page config
st.set_page_config(
    page_title="Yellow Sticky Trap Insect Detection",
    page_icon="🐝",
    layout="wide"
)

# Load class names from config (simplified for demo)
CLASS_NAMES = ['WF', 'MR', 'NC']  # Whitefly, Mealybug, and Mite

# Define the model path
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         "model", "best.pt")

# CSS to make the app look better
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.stButton>button {
    background-color: #ffeb3b;
    color: #333;
    font-weight: bold;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px 24px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.stButton>button:hover {
    background-color: #fdd835;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}
.upload-section {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.results-section {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
h1, h2, h3 {
    color: #333;
}
</style>
""", unsafe_allow_html=True)

def get_download_link(file_path, link_text):
    """Generate a download link for a file"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

def main():
    """Main function for Streamlit app"""
    
    # Header
    st.title("🐝 Yellow Sticky Trap Insect Detection")
    st.markdown("Upload an image of a yellow sticky trap to detect insects")
    
    # Sidebar with settings
    st.sidebar.title("Detection Settings")
    
    tile_size = st.sidebar.slider("Tile Size", 640, 1920, 1280, 128,
                                help="Size of tiles for processing large images")
    
    overlap = st.sidebar.slider("Overlap", 50, 200, 100, 10,
                               help="Overlap between adjacent tiles")
    
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05,
                                     help="Minimum confidence score for detections")
    
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.7, 0.05,
                                    help="IoU threshold for non-maximum suppression")
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📂 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process image if uploaded
    if uploaded_file is not None:
        # Create a temporary directory to store files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded image to a temporary file
            img_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(img_bytes))
            
            # Generate a unique filename for the temp image
            temp_img_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
            image.save(temp_img_path)
            
            # Display original image
            st.markdown('<div class="results-section">', unsafe_allow_html=True)
            st.subheader("📸 Original Image")
            st.image(image, caption="Original Upload", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Run detection when button is clicked
            if st.button("Run Detection"):
                # Show spinning progress indicator
                with st.spinner("Running detection... This may take a moment."):
                    try:
                        # Run inference
                        result_image, keep_indices, boxes, classes, scores = predict_on_tiled_image(
                            model_path=MODEL_PATH,
                            image_path=temp_img_path,
                            tile_size=tile_size,
                            overlap=overlap,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                            class_names=CLASS_NAMES
                        )
                        
                        # Save results to temporary files
                        result_img_path = os.path.join(temp_dir, f"result_{uuid.uuid4()}.jpg")
                        xml_path = os.path.join(temp_dir, f"annotation_{uuid.uuid4()}.xml")
                        
                        # Save result image
                        result_image.save(result_img_path)
                        
                        # Save XML annotation
                        save_annotation_as_xml(
                            temp_img_path,
                            xml_path,
                            boxes,
                            classes,
                            CLASS_NAMES
                        )
                        
                        # Display results
                        st.markdown('<div class="results-section">', unsafe_allow_html=True)
                        st.subheader("🔍 Detection Results")
                        
                        # Show detected objects count
                        st.markdown(f"### Found {len(boxes)} objects:")
                        
                        # Create a breakdown of detections by class
                        class_counts = {}
                        for cls in classes:
                            class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # Display class counts
                        for class_name, count in class_counts.items():
                            st.markdown(f"- **{class_name}**: {count}")
                        
                        # Display result image
                        st.image(result_image, caption="Detection Results", use_column_width=True)
                        
                        # Download links
                        st.subheader("📥 Download Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(get_download_link(result_img_path, "📷 Download Result Image"), unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(get_download_link(xml_path, "📝 Download XML Annotation"), unsafe_allow_html=True)
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error running detection: {str(e)}")

if __name__ == "__main__":
    main()
