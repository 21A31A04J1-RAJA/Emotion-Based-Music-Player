import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load your pre-trained emotion recognition model
emotion_model = load_model('emotion_model.h5')

# Load your pre-trained music retrieval model
# (For simplicity, we assume a dummy function for music retrieval)
def get_music_recommendations(emotion):
    # Dummy implementation; replace with actual music retrieval logic
    music_library = {
        'sadness': ['sad_song1.mp3', 'sad_song2.mp3'],
        'joy': ['happy_song1.mp3', 'happy_song2.mp3'],
        'anger': ['angry_song1.mp3', 'angry_song2.mp3'],
        'joy-anger': ['mixed_emotion_song1.mp3', 'mixed_emotion_song2.mp3'],
        'joy-surprise': ['surprise_song1.mp3', 'surprise_song2.mp3'],
        'joy-excitement': ['excited_song1.mp3', 'excited_song2.mp3'],
        'sad-anger': ['sad_anger_song1.mp3', 'sad_anger_song2.mp3'],
    }
    return music_library.get(emotion, [])

# Function to predict emotion from audio
def predict_emotion(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract features (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    features = np.mean(mfccs.T, axis=0)

    # Make emotion prediction
    emotion = emotion_model.predict(np.expand_dims(features, axis=0))
    emotion_label = np.argmax(emotion)
    
    # Convert label to emotion string (dummy example)
    labels = ['sadness', 'joy', 'anger', 'joy-anger', 'joy-surprise', 'joy-excitement', 'sad-anger']
    predicted_emotion = labels[emotion_label]
    
    return predicted_emotion

# Main function to generate playlist
def generate_playlist(user_audio_file):
    emotion = predict_emotion(user_audio_file)
    print(f"Detected emotion: {emotion}")
    
    # Get music recommendations based on emotion
    playlist = get_music_recommendations(emotion)
    print(f"Recommended playlist: {playlist}")

# Example usage
if __name__ == "__main__":
    user_audio_file = 'user_input_audio.mp3'  # Replace with the path to user's audio file
    generate_playlist(user_audio_file)
