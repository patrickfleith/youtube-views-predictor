from getpass import GetPassWarning
import streamlit as st
from PIL.Image import open

#from nltk.corpus import stopwords

# from nltk import word_tokenize

import pandas as pd

import numpy as np

import tensorflow as tf

import joblib


st.set_page_config(layout="wide")
st.title("Youtube Views Predictor")

nlp_scaler  = joblib.load("nlp_scaler")
num_scaler  = joblib.load("num_scaler")
tfidf       = joblib.load("nlp_tfidf")

def create_inference_data(weekday, subs, duration, thumbnail_path, description, title):

    cols = ['video_id', 'title', 'image_path', 'pdate', 'weekday', 'rdate', 'views','duration', 'subs', 'description']
    
    def convert_weekday(user_text="lundi"):

        daysOfWeek = {
            "lundi": 1.0,
            "mardi": 2.0,
            "mercredi": 3.0,
            "jeudi": 4.0,
            "vendredi": 5.0,
            "samedi": 6.0,
            "dimanche": 7.0}

        # try:
        #     val = daysOfWeek[user_text]
        # except:
        #     val = 1.0

        return daysOfWeek[user_text]

    data = [[0, title, thumbnail_path, np.nan, convert_weekday(weekday), np.nan, np.nan, duration, subs, description]]
    df_inference = pd.DataFrame(data=data, columns=cols)

    return df_inference

def nlp_processing_pipeline_inference(df_inference, tfidf, scaler):

    inference_corpus = (df_inference['title']+df_inference['description'].astype(str)).tolist()
    X_inference = tfidf.transform(inference_corpus)
    X_inference = X_inference.toarray()
    X_inference_scaled = scaler.transform(X_inference)
    return X_inference_scaled

def numerical_processing_pipeline_inference(df_inference, numerical_features, scaler):

    inference_num = df_inference[numerical_features]
    inference_num_scaled = scaler.transform(inference_num)
    return inference_num_scaled

#############################  SIDE BAR ##################

add_header = st.sidebar.header("Parameters")
add_selectbox = st.sidebar.selectbox("Select Youtuber Name",("Hugo Lisoir",))
SUBS = st.sidebar.number_input(label="Channel Subscribers", min_value=0, max_value=1000000, value=432000)
DURATION = st.sidebar.number_input(label="Duration (days)", min_value=30, max_value=300, value=30)

col1, col2 = st.columns([2, 1])

##########################################################

with col1:

    TITLE = st.text_input(label="Type video title")

    WEEKDAY = st.selectbox('Which day of the week are you publishing?', ('Monday', 'Tuesday', 'Wednesday', "Thursday", "Friday", "Saturday", "Sunday"))
    def convert_eng_to_fr_day(weekday):
        conv_dict = {"Monday": "lundi",
                    "Tuesday": "mardi",
                    "Wednesday": "mercredi",
                    "Thursday": "jeudi",
                    "Friday": "vendredi",
                    "Saturday": "samedi",
                    "Sunday": "dimanche"}
        return conv_dict[weekday]
    WEEKDAY = convert_eng_to_fr_day(WEEKDAY)

    uploaded_file = st.file_uploader(label="Upload the thumbnail", type=['png', 'jpg',])
    if uploaded_file is not None:

        img = open(uploaded_file)
        
        height = 124
        width = 69
        n_channels = 3
        all_img_array = np.zeros((1, height, width, n_channels))
        # streamlit
        image_resized = img.resize((height, width))
        img_array = np.swapaxes(np.asarray(image_resized), 0, 1)

        all_img_array[0, :] = img_array/255

        # display image of thumbnail to the app
        img = np.array(img)
        st.image(img)

    TEXT = st.text_input(label="Type here the text that appear on the thumbnail")

    if st.button('PREDICT'):

        # Prepare inputs for DNN model
        df_inference = create_inference_data(weekday=WEEKDAY, 
                                            subs=SUBS, 
                                            duration=DURATION, 
                                            thumbnail_path=".", 
                                            description=TEXT,
                                            title=TITLE)

        inference_img = all_img_array # IMG INPUTS
        
        inference_nlp = nlp_processing_pipeline_inference(df_inference=df_inference, tfidf=tfidf, scaler=nlp_scaler)
        numerical_features = ["weekday", "duration", "subs"]
        inference_num = numerical_processing_pipeline_inference(df_inference=df_inference, numerical_features=numerical_features, scaler=num_scaler)

        model = tf.keras.models.load_model('bynaris.h5')
        prediction = model.predict(x=[inference_nlp, inference_num, inference_img], batch_size=1)
        pred = round(prediction[0][0]/1e3)

        NOPE_MEH_BOUND = 100000
        MEH_OK_BOUND = 110000
        OK_GOOD_BOUND = 120000
        GOOD_GOD_BOUND = 140000

        lb = round(pred*0.7)
        ub = round(pred*1.3)
        
        st.write(f"PREDICTION: **{pred}** k views after {DURATION} days")

with col2:

        try:
            if prediction<NOPE_MEH_BOUND:
                st.image('app_img/img_nope.png')
            elif prediction<MEH_OK_BOUND:
                st.image('app_img/img_meh.png')
            elif prediction<OK_GOOD_BOUND:
                st.image('app_img/img_ok.png')
            elif prediction<GOOD_GOD_BOUND:
                st.image('app_img/img_great.png')
            elif prediction>GOOD_GOD_BOUND:
                st.image('app_img/img_god.png')

        except: 
            pass
        