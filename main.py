import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from pathlib import Path


path_to_model = Path('./trained_plant_disease_model.keras')

# Replace with your Gemini API key
def get_answer(question):
    url = "https://open-ai21.p.rapidapi.com/chatgpt"

    payload = {
	 "messages": [
		{
			"role": "user",
			"content": [{"text": question}]
		}
	 ],
	 "web_access": False
    }
    headers = {
	 "content-type": "application/json",
	 "X-RapidAPI-Key": "7b11c23847msh7a5d7127194ac43p12b273jsn2ce1c9d969d0",
	 "X-RapidAPI-Host": "open-ai21.p.rapidapi.com"
    }

    response = requests.post(url, json=payload, headers=headers)
    generated_text = response.json()['result']
    return generated_text

# News API configuration
NEWS_API_KEY = 'd64b3bcfb79e4b53b7f587b3cd1ed688'  # Replace 'YOUR_NEWS_API_KEY' with your actual News API key
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'

# Function to fetch top 5 news articles related to farmers
def fetch_news():
    url = f'https://newsapi.org/v2/everything?q=farmers&apiKey=d64b3bcfb79e4b53b7f587b3cd1ed688&pageSize=5'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        return None

def get_current_price(symbol):
    url = f'https://api.gemini.com/v1/pubticker/{symbol}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
# Tensorflow Model Prediction
def model_prediction(test_image,path_to_model):
    model = tf.keras.models.load_model(path_to_model)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Page styling
def set_page_style():
    st.markdown(
        """
        <style>
            .reportview-container {
                background: linear-gradient(to bottom, #e6f7ff, #ffffff);
            }
            .sidebar .sidebar-content {
                background: linear-gradient(to bottom, #334d5c, #4b676d);
                color: white;
            }
            .Widget>label {
                color: #4b676d;
            }
            .stButton>button {
                color: white;
                background-color: #4b676d;
            }
            .stButton>button:hover {
                background-color: #3c5a66;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_page_style()

# Navigation bar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    
   
    st.markdown(
        """
        Welcome to the Plant Disease Recognition System! 🌿🔍

        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        
        """
    )
     # Display top 5 news articles related to farmers


    st.subheader("Top 5 News Articles Related to Farmers")
    
    news_data = fetch_news()
    if news_data:
     for i in range(0, len(news_data), 2):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {news_data[i]['title']}")
            st.write(news_data[i]['description'])
            st.write(f"Source: {news_data[i]['source']['name']}")
            if news_data[i]['urlToImage']:  # Check if image is available
                st.image(news_data[i]['urlToImage'], caption='Image', use_column_width=True)
              # Append a unique identifier to the button label
            st.write(f"[Read more]({news_data[i]['url']})")
            st.write("---")
        if i + 1 < len(news_data):
            with col2:
                st.markdown(f"### {news_data[i + 1]['title']}")
                st.write(news_data[i + 1]['description'])
                st.write(f"Source: {news_data[i + 1]['source']['name']}")
                if news_data[i + 1]['urlToImage']:  # Check if image is available
                    st.image(news_data[i + 1]['urlToImage'], caption='Image', use_column_width=True)
                 # Append a unique identifier to the button label
                st.write(f"[Read more]({news_data[i + 1]['url']})")
                st.write("---")



# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown(
        """
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
        This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
        A new directory containing 33 test images is created later for prediction purposes.
        #### Content
        1. train (70295 images)
        2. test (33 images)
        3. validation (17572 images)
        """
    )

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image is not None:
        st.image(test_image, width=300, caption="Uploaded Image")

    # Predict button
    if st.button("Predict"):
        st.spinner(text="Predicting...")
        result_index = model_prediction(test_image,path_to_model)
        class_name = [
            "Apple___Apple_scab",
            "Apple___Black_rot",
            "Apple___Cedar_apple_rust",
            "Apple___healthy",
            "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew",
            "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
            "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)",
            "Peach___Bacterial_spot",
            "Peach___healthy",
            "Pepper,_bell___Bacterial_spot",
            "Pepper,_bell___healthy",
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
            "Raspberry___healthy",
            "Soybean___healthy",
            "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch",
            "Strawberry___healthy",
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy",
        ]
        st.success("Model predicts it's {}".format(class_name[result_index]))
        st.title('Gemini API Example')
        
        question = f"I have a plant with a {class_name[result_index]}. How do I treat it? Answer in 10 words"

        answer = get_answer(question)

        if answer:
          st.success(f"Answer: {answer}")
        else:
          st.error("No valid response received.")



        if answer:
          st.success(f"Answer: {answer}")
        else:
          st.error("No valid response received.")
