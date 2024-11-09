import streamlit as st
from ytmusicapi import YTMusic
from fer import FER
import cv2
import random
import numpy as np
from collections import Counter
import time

# Initialize the YTMusic API and FER Emotion Detector
ytmusic = YTMusic()
detector = FER()

# Emotion-to-Genre Mapping
emotion_to_genre = {
    'happy': 'upbeat Punjabi songs',
    'sad': 'sad Punjabi songs',
    'angry': 'Punjabi rock songs',
    'neutral': 'popular Punjabi songs',
    'surprise': 'trending Punjabi songs',
    'fear': 'inspirational Punjabi songs',
    'disgust': 'soothing Punjabi songs'
}

# Track recently recommended songs to avoid duplicates
recently_recommended = set()

# Define a function to get song recommendations based on emotion
def get_songs_for_emotion(emotion, num_songs=5):
    genre_query = emotion_to_genre.get(emotion, 'hindi songs')
    results = ytmusic.search(genre_query, filter="songs")
    
    # Shuffle results to get a variety of songs
    random.shuffle(results)
    
    playlist = []
    for song in results:
        if song['videoId'] not in recently_recommended:
            playlist.append({
                'title': song['title'],
                'artist': song['artists'][0]['name'],
                'videoId': song['videoId'],
                'thumbnail': song['thumbnails'][0]['url']
            })
            recently_recommended.add(song['videoId'])

            # Limit the cache to avoid memory bloat
            if len(recently_recommended) > 20:
                recently_recommended.pop()

            # Stop once we have the desired number of songs
            if len(playlist) == num_songs:
                break

    # Return the playlist
    return playlist

# Streamlit App
st.title("Face-Based Emotion Detection Music Recommender")

# Start detection button
if st.button("Start Detection"):
    st.write("Initializing camera. Please wait a few seconds...")

    # Add a delay to allow the camera to initialize
    time.sleep(3)  # Adjust this delay as needed

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Unable to access the webcam.")
    else:
        stop = st.button("Stop Detection")
        
        # Capture frames and perform detection
        while not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            faces = detector.detect_emotions(rgb_frame)
            if faces:
                # Only proceed if a face is detected
                face_emotion = faces[0]  # Use the first detected face

                # Get the dominant emotion for this face
                dominant_emotion = max(face_emotion['emotions'], key=face_emotion['emotions'].get)
                
                # Display the detected face and emotion in Streamlit
                x, y, w, h = face_emotion['box']
                face_frame = rgb_frame[y:y+h, x:x+w]
                st.image(face_frame, caption=f"Detected Emotion: {dominant_emotion.capitalize()}", width=200)
                
                # Provide song recommendations based on the detected emotion
                recommended_songs = get_songs_for_emotion(dominant_emotion)
                st.write(f"### Recommended Songs for {dominant_emotion.capitalize()}:")

                for song in recommended_songs:
                    st.write(f"**{song['title']}** by {song['artist']}")
                    st.image(song['thumbnail'], width=100)
                    song_url = f"https://www.youtube.com/watch?v={song['videoId']}"
                    st.write(f"[Listen on YouTube]({song_url})")
                
                # Clear recommendations and end the loop after displaying songs
                stop = True
            
            # Add a delay between frames to reduce load
            time.sleep(0.5)

        # Release video capture
        cap.release()
        cv2.destroyAllWindows()
        st.write("Detection stopped.")
else:
    st.write("Press 'Start Detection' to begin face detection.")
