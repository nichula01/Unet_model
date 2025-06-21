import streamlit as st
import os
import torch
import numpy as np
import tifffile
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# We need to import the U-Net and some functions from the training script
# Make sure this 'app.py' is in the same folder as 'run_pipeline.py'
from run_pipeline import UNet, get_transforms, normalize_for_display, run_inference_on_image, config, DEVICE

# =======================================================================================
# Streamlit App UI
# =======================================================================================

st.set_page_config(layout="wide")
st.title("🛰️ Slum Detection Interface")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = os.path.join(config.MODEL_DIR, f"{config.MODEL_NAME}_{config.TARGET_CITY}_best.pth")
    if not os.path.exists(model_path):
        st.error(f"Model file not found! Please train the model first by running `run_pipeline.py`. Expected path: {model_path}")
        return None
    
    model = UNet(n_channels=3, n_classes=config.CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- File Uploader ---
st.header("1. Upload Satellite Image")
uploaded_file = st.file_uploader("Choose a .tif file...", type=["tif", "tiff"])

if uploaded_file is not None and model is not None:
    # Read uploaded image
    try:
        image = tifffile.imread(uploaded_file)
        # Ensure it's a 3-channel image
        if len(image.shape) == 2: # Grayscale
             image = np.stack([image]*3, axis=-1)
        elif image.shape[2] > 3: # More than 3 channels
             image = image[:, :, :3]
    except Exception as e:
        st.error(f"Error reading the image file: {e}")
        image = None

    if image is not None:
        st.header("2. Run Detection")
        
        # Display uploaded image
        st.subheader("Your Uploaded Image")
        st.image(normalize_for_display(image), caption="Satellite Image", use_column_width=True)

        if st.button("Detect Slums", use_container_width=True):
            with st.spinner("Model is running inference... This may take a moment."):
                # Get the same validation transform used in training
                _, val_transform = get_transforms()
                
                # Run the inference function
                predicted_mask = run_inference_on_image(image, model, val_transform)
                
                # Create the overlay visualization
                rgb_display = normalize_for_display(image)
                cmap = ListedColormap(['#00000000', 'gray', 'red']) # Transparent, Gray, Red
                
                fig, ax = plt.subplots(figsize=(12, 12))
                ax.imshow(rgb_display)
                ax.imshow(predicted_mask, cmap=cmap, alpha=0.6)
                ax.axis('off')
                
                # Save the figure to a buffer to display in Streamlit
                from io import BytesIO
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                
                st.header("3. Detection Result")
                st.image(buf, caption="Predicted Slum Segmentation", use_column_width=True)
                
                # Allow downloading the result
                st.download_button(
                   label="Download Result Image",
                   data=buf,
                   file_name="detection_result.png",
                   mime="image/png"
                )