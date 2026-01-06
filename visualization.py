import streamlit as st
from PIL import Image
import os

def display_image_with_title(image_path, title, width=800):
    """Display an image with a title if the file exists"""
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            # Resize image to specified width while maintaining aspect ratio
            aspect_ratio = img.height / img.width
            new_height = int(width * aspect_ratio)
            img_resized = img.resize((width, new_height), Image.Resampling.LANCZOS)
            
            st.markdown(f'<h3 style="color: #ff2546; font-family: Montserrat, sans-serif;">{title}</h3>', unsafe_allow_html=True)
            st.image(img_resized, use_container_width=False)
        except Exception as e:
            st.error(f"Error loading image {image_path}: {e}")
    else:
        st.warning(f"Image not found: {image_path}")

def app():
    st.markdown('<h1 class="title-font">📊 Visualize and Analyze Model\'s Performance</h1>', unsafe_allow_html=True)
    st.markdown("""
    **🔍 Dive into the details of our model's performance:**

    - **Test Accuracy:** Check the final accuracy on the test dataset. 🎯
    - **Training and Validation Accuracy:** See how the model improves with each epoch. 📈
    - **Loss Metrics:** Track the progress of loss reduction throughout training. 📉
    - **Confusion Matrix:** Understand the model's classification results with a visual confusion matrix. 🧩
    """)
    
    st.markdown('<h2 class="sub-title">AI Image Detector Model Performance</h2>', unsafe_allow_html=True)
    
    with st.expander("See More Details"):
        st.markdown("""
                    ### 📂 Model Details:
                    - **Architecture:** Convolutional Neural Network (CNN) 🧠
                    - **Input Size:** 256x256 pixels 📐
                    - **Classes:** Real vs Fake (AI-Generated) 🖼️
                    - **Training Techniques:** Data augmentation, dropout, batch normalization 🔧
                    
                    ### 🧠 How It Helps:

                    - **Gain Insights:** Identify trends and performance patterns in model training. 🌟
                    - **Analyze Trends:** Detect overfitting or underfitting issues. 🔍
                    - **Enhance Performance:** Use insights to fine-tune hyperparameters and improve accuracy. ⚙️
                    - **Interact with Visuals:** Explore the training history and confusion matrix visualizations. 📊
                    """)
    
    # Display training accuracy and loss images
    st.markdown('<div class="dmain">', unsafe_allow_html=True)
    
    # Training History Images
    display_image_with_title('images/training_accuracy.png', '📈 Training and Validation Accuracy')
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="dmain">', unsafe_allow_html=True)
    
    display_image_with_title('images/training_loss.png', '📉 Training and Validation Loss')
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Confusion Matrix Image
    st.markdown('<div class="dmain">', unsafe_allow_html=True)
    
    display_image_with_title('images/confusion_matrix.png', '🧩 Confusion Matrix')
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional metrics section
    st.markdown('<div class="dmain">', unsafe_allow_html=True)
    st.markdown("### 📈 Model Performance Summary")
    
    st.markdown("""
        <p style='font-family: Montserrat, sans-serif; font-size: 16px;'>
        The model has been trained to distinguish between real photographs and AI-generated images.
        The visualizations above show the training progress, validation performance, and final classification results.
        </p>
    """, unsafe_allow_html=True)
    
    # You can add manual metrics here if you know them
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Accuracy", "85.04%", "")
    
    with col2:
        st.metric("Precision", "83.7%", "")
    
    with col3:
        st.metric("Recall", "83.7%", "")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions for users
    st.info("""
        📁 **Note:** Place your training visualization images in the `images/` folder:
        - `images/training_accuracy.png` - Training and validation accuracy plot
        - `images/training_loss.png` - Training and validation loss plot
        - `images/confusion_matrix.png` - Confusion matrix visualization
    """)

if __name__ == "__main__":
    app()