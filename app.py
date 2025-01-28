import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.neighbors import NearestNeighbors


try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    try:
        model = joblib.load("model.pkl")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


try:
    tracks_df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    st.error("The dataset file 'dataset.csv' is missing. Please upload it.")
    st.stop()

features = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]

if not all(feature in tracks_df.columns for feature in features):
    missing_features = [feature for feature in features if feature not in tracks_df.columns]
    st.error(f"The dataset is missing these required features: {missing_features}")
    st.stop()


if 'track_name' in tracks_df.columns:
    le = LabelEncoder()
    tracks_df['encoded_track'] = le.fit_transform(tracks_df['track_name'])
else:
    st.error("The dataset must contain a 'track_name' column.")
    st.stop()


scaler = StandardScaler()
data_scaled = scaler.fit_transform(tracks_df[features])

st.set_page_config(page_title="Spotify Recommendation App", layout="wide")


st.markdown(
    """
    <style>
    .main {
        background-color: #121212;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stButton > button {
        background-color: #1DB954;
        color: white !important;
        border: None;
        font-size: 16px;
        border-radius: 25px;
        padding: 8px 20px;
        text-align: center;
    }
    .stButton > button:hover {
        background-color: #1ed760;
    }
    .spotify-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
    }
    .spotify-logo {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80px;
        margin-bottom: 10px;
    }
    .stSelectbox label, .stSlider label, .stSubheader, .stText {
        text-align: center;
        display: block;
        
    }
    label[data-testid="stSelectboxLabel"], label[data-testid="stSliderLabel"] {
        font-size: 18px !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
    <img class="spotify-logo" src="https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg">
    <div class="spotify-title">Spotify Music Recommendation</div>
""", unsafe_allow_html=True)


st.markdown('<h4 style="text-align:center; " >Select a Track</h4>', unsafe_allow_html=True)


selected_track = st.selectbox(
    "Scroll to select a track from our playlist:",
    tracks_df['track_name'].values
)



num_recommendations = st.slider(
    "How many songs would you like to be suggested?",
    min_value=1,
    max_value=15,
    value=10  
)


st.markdown(f'<p style="text-align:center; font-size:18px;">You selected: <b>{selected_track}</b></p>', unsafe_allow_html=True)

if st.button("Get Recommendations"):
    try:
        
        selected_track_index = tracks_df[tracks_df['track_name'] == selected_track].index[0]

        
        selected_encoded = data_scaled[selected_track_index].reshape(1, -1)  # Reshape for model input

        
        distances, indices = model.kneighbors(selected_encoded, n_neighbors=num_recommendations + 1)
        recommended_indices = indices[0][1:]  # Exclude the first index (selected track itself)

        
        st.subheader("Recommended Tracks:")
        for i, track_index in enumerate(recommended_indices):
            track_name = tracks_df['track_name'].iloc[track_index]
            st.markdown(f"<p style='text-align:center; font-size:16px;'> {i+1}. {track_name}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred while predicting recommendations: {e}")
