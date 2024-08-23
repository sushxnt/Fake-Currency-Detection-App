import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras._tf_keras.keras.models import load_model

         
# Load fraud detection model
fraud_detection_model = load_model('testmodelBN.h5')



# Function to preprocess the image before feeding it to the detection model
def preprocess_detection_image(image):
    try:
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image,axis=0)
        # if image.shape[-1]==1:
        #     image=np.repeat(image,3,axis=-1)
        # elif image.shape[-1]==4:
        #     image=image[...,:3]
        #Reshape to (1,224,224,3)
        image=image.reshape((1,224,224,3))        
        return image
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to detect fraud
def detect_fraud(img):
    if img is None:
        raise ValueError("Invalid image input")
    prediction = fraud_detection_model.predict(img)
    prediction=prediction>0.5
    print(prediction)
    label = int(prediction[0][0])
    print(label)
    # label=prediction
    if label == 0:
        return 'Fake'
    if label == 1:
        return 'Genuine'
    return 'Unknown'

# Custom CSS for sticky navigation bar
st.markdown("""
    <style>
    .navbar {
        display: flex;
        justify-content: center;
        background-color: #343a40;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .navbar h1 {
        margin: 0;
        font-size: 1.5rem;
        font-family:Monospace;
        color: white;
    }
    .content {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
    }
    .image-preview {
        margin-bottom: 20px;
    }
    .analyze-button {
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        border-top: 1px solid #e1e1e1;
        padding: 10px;
    }
    </style>
    <div class="navbar">
        <h1>💵 Fake Nepali Currency Detection</h1>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("<div class='content'>", unsafe_allow_html=True)
st.write("### Upload an image for detection:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"], help="Upload a JPG or JPEG image of the currency note.")

if uploaded_file is not None:
    st.markdown("<div class='image-preview'>", unsafe_allow_html=True)
    #  mage preview
    st.image(uploaded_file, caption="Uploaded Image", width=100)
    st.markdown("</div>", unsafe_allow_html=True)

    # st.markdown("<hr style='border:1px solid gray; width:80%; margin:auto;'>", unsafe_allow_html=True)

    st.markdown("<div class='analyze-button'>", unsafe_allow_html=True)
    analyze_button = st.button("Analyze Image")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze_button:
        with st.spinner('Analyzing the image...'):
            fraud_detection_img = preprocess_detection_image(Image.open(uploaded_file))
            if fraud_detection_img is not None:
                fraud_result = detect_fraud(fraud_detection_img)
                if fraud_result == 'Genuine':
                    st.success(f"🎉 The currency note is: **{fraud_result}**")
                    #load and display fake.html content
                    with open('real.html', 'r',encoding="utf-8") as file:
                        html_content = file.read()

                    st.components.v1.html(html_content, height=100)
                else:
                    st.error(f"🚨 The currency note is: **{fraud_result}**")
                    #load and display fake.html content
                    with open('fake.html', 'r',encoding="utf-8") as file:
                        html_content = file.read()

                    st.components.v1.html(html_content, height=100)
                    

            else:
                st.error("Image preprocessing failed. Cannot proceed with fraud detection.")
else:
    st.info("Please upload an image file to proceed.")

# Close the content div
st.markdown("</div>", unsafe_allow_html=True)

# footer
st.markdown("""
    <div class="footer">
        <p>© 2024 Fake Nepali Currency Detection</p>
    </div>
    """, unsafe_allow_html=True)