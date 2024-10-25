import streamlit as st
import tensorflow as tf
import numpy as np
import streamlit.components.v1 as components
import requests
from pathlib import Path
from dotenv import load_dotenv
import os
import google.generativeai as genai
import requests
from requests.exceptions import ConnectionError

def get_geolocation():
  """
  Fetches user's location data using an IP geolocation API.

  Returns:
      A dictionary containing location data or None if there's an error.
  """
  key = "0ef9739fc1c44f209980016108e89f94"  
  url = "https://api.ipgeolocation.io/ipgeo?apiKey=" + key
  response = requests.get(url)
  if response.status_code == 200:
    return response.json()
  else:
    return None

def get_weather_data(latitude, longitude, api_key):
  """
  Fetches weather data from OpenWeatherMap API for given coordinates and API key.

  Args:
      latitude: User's latitude.
      longitude: User's longitude.
      api_key: OpenWeatherMap API key.

  Returns:
      A dictionary containing weather data or None if there's an error.
  """
  url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}"
  response = requests.get(url)
  if response.status_code == 200:
    return response.json()
  else:
    return None

def display_weather(weather_data):
  """
  Displays weather information on the Streamlit app.

  Args:
      weather_data: A dictionary containing weather data.
  """
  if weather_data:
    city = weather_data["name"]
    temperature = kelvin_to_celsius(weather_data["main"]["temp"])
    humidity = weather_data["main"]["humidity"]
    description = weather_data["weather"][0]["description"]

    # Create columns for layout
    col1, col2, col3 = st.columns([2, 4, 2])

    # Display city with globe emoji
    with col1:
      st.write("Your City: ", city)

    # Display temperature with sun/cloud emoji
    with col2:
      if "cloud" in description:
        st.write("⛅ Temperature: ", f"{temperature:.2f} °C")
      else:
        st.write("☀️ Temperature: ", f"{temperature:.2f} °C")

    # Display humidity with water droplet emoji
    with col3:
      st.write("Humidity: ", f"{humidity}%")
  else:
    st.error("Error fetching weather data")

def kelvin_to_celsius(kelvin):
  """Converts temperature from Kelvin to Celsius."""
  return kelvin - 273.15


st.markdown(
    """
    <style>
        /* Add some animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .fadeIn {
            animation-name: fadeIn;
            animation-duration: 1s;
        }
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
        .fadeOut {
            animation-name: fadeOut;
            animation-duration: 1s;
        }

        /* Light mode */
        body[data-theme="light"] {
            --background-color: #ffffff;
            --text-color: #333333;
            --button-bg-color: #4b676d;
            --button-text-color: #ffffff;
            --button-hover-bg-color: #3c5a66;
        }
        /* Dark mode */
        body[data-theme="dark"] {
            --background-color: #1e1e1e;
            --text-color: #ffffff;
            --button-bg-color: #334d5c;
            --button-text-color: #ffffff;
            --button-hover-bg-color: #3c5a66;
        }
        body {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        .stButton>button {
            color: var(--button-text-color);
            background-color: var(--button-bg-color);
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: var(--button-hover-bg-color);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

path_to_model = Path('trained_plant_disease_model.h5')

genai.configure(api_key="AIzaSyB3QUjUKd8Hv0yZOFbbg5l29QgfUfIsuFI")
model = genai.GenerativeModel('gemini-pro')

# Gemini uses 'model' for assistant; Streamlit uses 'assistant'
def role_to_streamlit(role):
  if role == "model":
    return "assistant"
  else:
    return role
# News API configuration
NEWS_API_KEY = 'f100001a7aef4d318dcb0625d83ba4b2'  # Replace 'YOUR_NEWS_API_KEY' with your actual News API key
NEWS_API_URL = 'https://newsapi.org/v2/top-headlines'

# Function to fetch top 5 news articles related to farmers
def fetch_news():
    url = f'https://newsapi.org/v2/everything?q=farmers&apiKey=f100001a7aef4d318dcb0625d83ba4b2&pageSize=5'
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

def convert_class_name(class_name):
    # Split the class name on underscores and capitalize each word
    words = class_name.split("_")
    converted_name = " ".join(word.capitalize() for word in words)
    return converted_name

# Page styling
def set_page_style():
    st.markdown(
        """
        <style>
            /* Add some animations */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            .fadeIn {
                animation-name: fadeIn;
                animation-duration: 1s;
            }
            @keyframes fadeOut {
                from { opacity: 1; }
                to { opacity: 0; }
            }
            .fadeOut {
                animation-name: fadeOut;
                animation-duration: 1s;
            }

            /* Light mode */
            body[data-theme="light"] {
                --background-color: #ffffff;
                --text-color: #333333;
                --button-bg-color: #4b676d;
                --button-text-color: #ffffff;
                --button-hover-bg-color: #3c5a66;
            }
            /* Dark mode */
            body[data-theme="dark"] {
                --background-color: #1e1e1e;
                --text-color: #ffffff;
                --button-bg-color: #334d5c;
                --button-text-color: #ffffff;
                --button-hover-bg-color: #3c5a66;
            }
            body {
                background-color: var(--background-color);
                color: var(--text-color);
            }
            .stButton>button {
                color: var(--button-text-color);
                background-color: var(--button-bg-color);
                transition: background-color 0.3s;
            }
            .stButton>button:hover {
                background-color: var(--button-hover-bg-color);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
set_page_style()

# Navigation bar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "About", "Disease Recognition"])
def set_language():
    language = st.sidebar.selectbox("Select Language", ("English", "Hindi"))
    return language
# Main Page
if app_mode == "Home":
    language = set_language()
    if language == "English":
        st.header("PLANT DISEASE RECOGNITION SYSTEM")
        
        with st.spinner("Locating you..."):
            # Get user location
            response = get_geolocation()
            if not response:
                st.error("Unable to determine your location. Please try again later.")

            latitude = response["latitude"]
            longitude = response["longitude"]

            # Get weather data
            api_key = "1bcea20641a026ba8378e6f5fec4c71f"  # Replace with your OpenWeatherMap API key
            weather_data = get_weather_data(latitude, longitude, api_key)

            # Display weather or error message
            display_weather(weather_data)  
        
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

    elif language == "Hindi":
        st.header("वनस्पति रोग पहचान प्रणाली")
        with st.spinner("आपकी स्थिति का पता लगाया जा रहा है..."):
            # उपयोगकर्ता स्थान प्राप्त करें
            response = get_geolocation()
            if not response:
                st.error("आपकी स्थिति का निर्धारण नहीं किया जा सका। कृपया बाद में पुनः प्रयास करें।")

            latitude = response["latitude"]
            longitude = response["longitude"]

            # मौसम डेटा प्राप्त करें
            api_key = "1bcea20641a026ba8378e6f5fec4c71f"  # अपनी OpenWeatherMap API कुंजी के साथ बदलें
            weather_data = get_weather_data(latitude, longitude, api_key)

            # मौसम या त्रुटि संदेश दिखाएं
            display_weather(weather_data)  

        st.markdown(
            """
            वनस्पति रोग पहचान प्रणाली में आपका स्वागत है! 🌿🔍

            हमारा उद्देश्य वनस्पति रोगों की पहचान करने में सहायक होना है। किसी पौधे की तस्वीर अपलोड करें, और हमारी प्रणाली उसे विशेष रूप से पहचानने के लिए विश्लेषित करेगी। हम साथ मिलकर हमारे फसलों की सुरक्षा करें और एक स्वस्थ फसल की सुनिश्चित करें!

            ### यह कैसे काम करता है
            1. **तस्वीर अपलोड करें:** **विषाणु पहचान** पृष्ठ पर जाएं और संदिग्ध रोगों के साथ किसी पौधे की तस्वीर अपलोड करें।
            2.**विश्लेषण:** हमारी प्रणाली उन्नत एल्गोरिदम का उपयोग करके छवि को प्रसंस्करण करेगी ताकि संभावित रोगों की पहचान की जा सके।
            3. **परिणाम:** परिणाम और आगे की कार्रवाई के लिए सिफारिशों को देखें।

            ### हमारे चुनाव क्यों?
            - **सटीकता:** हमारी प्रणाली सटीक रोग पहचान के लिए उन्नत मशीन लर्निंग तकनीकों का उपयोग करती है।
            - **उपयोगकर्ता मित्र योग्य:** सरल और स्पष्ट अंतरफलक के लिए सरल और सहज इंटरफ़ेस।
            - **तेज और दक्ष:** नतीजे सेकंडों में प्राप्त करें, जो त्वरित निर्णय लेने की अनुमति देता है।

   
            """
        )
        # किसानों से संबंधित शीर्ष 5 समाचार आलेखों को प्रदर्शित करें


        st.subheader("किसानों से संबंधित शीर्ष 5 समाचार आलेख")
    
        news_data = fetch_news()
        if news_data:
         for i in range(0, len(news_data), 2):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### {news_data[i]['title']}")
                st.write(news_data[i]['description'])
                st.write(f"Source: {news_data[i]['source']['name']}")
                if news_data[i]['urlToImage']:  # यदि छवि उपलब्ध है तो जाँचें
                    st.image(news_data[i]['urlToImage'], caption='Image', use_column_width=True)
                 # बटन लेबल को एकदमिक पहचानकर्ता में जोड़ें
                st.write(f"[अधिक पढ़ें]({news_data[i]['url']})")
                st.write("---")
            if i + 1 < len(news_data):
                with col2:
                    st.markdown(f"### {news_data[i + 1]['title']}")
                    st.write(news_data[i + 1]['description'])
                    st.write(f"Source: {news_data[i + 1]['source']['name']}")
                    if news_data[i + 1]['urlToImage']:  # यदि छवि उपलब्ध है तो जाँचें
                        st.image(news_data[i + 1]['urlToImage'], caption='Image', use_column_width=True)
                     # बटन लेबल को एकदमिक पहचानकर्ता में जोड़ें
                    st.write(f"[अधिक पढ़ें]({news_data[i + 1]['url']})")
                    st.write("---")





# About Project
elif app_mode == "About":
    language = set_language()
    if language == "English":
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
    elif language == "Hindi":
        st.header("विषय")
        st.markdown(
            """
            #### डेटासेट के बारे में
            यह डेटासेट मूल डेटासेट से ऑफलाइन वृद्धि का उपयोग करके पुनः बनाया गया है। मूल डेटासेट इस GitHub रेपो पर मिल सकता है।
            इस डेटासेट में लगभग 87K RGB छवियां हैं जो स्वस्थ और बीमारी प्रदान करती हैं जो 38 विभिन्न वर्गों में वर्गीकृत है। कुल डेटासेट को डायरेक्ट्री संरचना को संरक्षित रखते हुए 80/20 अनुपात में प्रशिक्षण और मान्यता सेट में विभाजित किया गया है।
            पूर्वानुमान के उद्देश्य से बाद में 33 परीक्षण छवियों की एक नई निर्देशिका बनाई गई है।
            #### सामग्री
            1. प्रशिक्षण (70295 छवियां)
            2. परीक्षण (33 छवियां)
            3. मान्यता (17572 छवियां)
            """
        )

# Prediction Page
elif app_mode == "Disease Recognition":
    language = set_language()
    if language == "English":
        response = get_geolocation()
        if not response:
                st.error("Unable to determine your location. Please try again later.")

        latitude = response["latitude"]
        longitude = response["longitude"]

            # Get weather data
        api_key = "1bcea20641a026ba8378e6f5fec4c71f"  # Replace with your OpenWeatherMap API key
        weather_data = get_weather_data(latitude, longitude, api_key)
        if weather_data:
         city = weather_data["name"]
         temperature = kelvin_to_celsius(weather_data["main"]["temp"])
         humidity = weather_data["main"]["humidity"]
         description = weather_data["weather"][0]["description"]
        st.header("Disease Recognition")
        st.write("Location: ",city)
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
          result=convert_class_name(class_name[result_index])
          st.success("Model predicts it's {}".format(result))
          prompt = f"What are the treatment options for {result} disease? I live in {city} with temperature of {temperature} and humidity {humidity}"
          if "chat" not in st.session_state:
            st.session_state.chat = model.start_chat(history=[])

        # Treatment prompt and response
          prompt = f"What are the treatment options for {result} disease? I live in {city} with temperature of {temperature} and humidity {humidity}"
          response = st.session_state.chat.send_message(prompt)

        # Display prediction and treatment info
          st.write("**Predicted Disease:**", result)
          st.write("**Possible Treatments from Assistant:**")
          with st.expander("See treatment details"):
            st.write(response.text)

        # Chat interaction section
        st.title("Chat with Plant AI!",city)
        if "chat" not in st.session_state:
            st.session_state.chat = model.start_chat(history=[])
                # Display chat history
        for message in st.session_state.chat.history:
            with st.chat_message(role_to_streamlit(message.role)):
                st.markdown(message.parts[0].text)

        # User input and response cycle
        if prompt := st.chat_input("I possess a well of knowledge. What would you like to know?"):
            st.chat_message("user").markdown(prompt)
            response = st.session_state.chat.send_message(prompt)
            with st.chat_message("assistant"):
                st.markdown(response.text)

    elif language == "Hindi":
        st.header("रोग पहचान")
        test_image = st.file_uploader("छवि चुनें:")
        if st.button("छवि दिखाएं") and test_image is not None:
            st.image(test_image, width=300, caption="अपलोड की गई छवि")

        # Predict button
        if st.button("पूर्वानुमान करें"):
            st.spinner(text="पूर्वानुमान किया जा रहा है...")
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
            result = convert_class_name(class_name[result_index])
            st.success("मॉडल यह पूर्वानुमान करता है कि यह {}".format(result))
            prompt = f"{result} रोग के लिए उपचार विकल्प क्या हैं?"
            if "chat" not in st.session_state:
                st.session_state.chat = model.start_chat(history=[])

            # उपचार पूछें और प्रतिक्रिया
            response = st.session_state.chat.send_message(prompt)

            # पूर्वानुमान और उपचार जानकारी प्रदर्शित करें
            st.write("**पूर्वानुमानित रोग:**", result)
            st.write("**सहायक से संभावित उपचार:**")
            with st.expander("उपचार विवरण देखें"):
                st.write(response.text)

            # चैट परिचय खंड
        st.title("Plant AI के साथ चैट करें!")
        if "chat" not in st.session_state:
                st.session_state.chat = model.start_chat(history=[])
            # चैट इतिहास प्रदर्शित करें
        for message in st.session_state.chat.history:
                with st.chat_message(role_to_streamlit(message.role)):
                    st.markdown(message.parts[0].text)

            # उपयोगकर्ता इनपुट और प्रतिक्रिया चक्र
        if prompt := st.chat_input("मेरे पास ज्ञान का अच्छा भंडार है। आप क्या जानना चाहेंगे?"):
                st.chat_message("user").markdown(prompt)
                response = st.session_state.chat.send_message(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response.text)
