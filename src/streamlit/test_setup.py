"""
Test script to verify the Streamlit app setup
"""
import os
import streamlit as st

def main():
    st.title("Setup Test")
    st.write("If you can see this message, the Streamlit app is properly set up.")
    
    # Check if the model exists
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "model", "best.pt")
    
    if os.path.exists(model_path):
        st.success(f"✅ Model found at: {model_path}")
    else:
        st.error(f"❌ Model not found at: {model_path}")
    
    # Check import paths
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.inference import predict_on_tiled_image
        st.success("✅ Successfully imported inference module")
    except Exception as e:
        st.error(f"❌ Failed to import inference module: {str(e)}")
    
if __name__ == "__main__":
    main()
