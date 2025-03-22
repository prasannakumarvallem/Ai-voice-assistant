from playsound import playsound
import eel
from engine.command import speak
from engine.config import ASSISTANT_NAME
import os
import pywhatkit as kit
import re
from hugchat import hugchat
import librosa
import numpy as np
import io
import soundfile as sf

# Initialize HugChat once
chatbot = hugchat.ChatBot(cookie_path="engine\\cookies.json")  
conversation_id = chatbot.new_conversation()  
chatbot.change_conversation(conversation_id)  

@eel.expose
def playAssistantSound():
    """ Play assistant startup sound """
    music_dir = 'www\\assets\\audio\\start_sound.mp3'
    playsound(music_dir)

def openCommand(query):
    """ Opens applications based on user command. """
    query = query.replace(ASSISTANT_NAME, "").replace("open", "").strip().lower()

    if query:
        speak("Opening " + query)
        os.system(f'start {query}')
    else:
        speak("Application not found")

def PlayYoutube(query):
    """ Plays a YouTube video based on the user’s command. """
    search_term = extract_yt_term(query)
    speak("Playing " + search_term + " on YouTube")
    kit.playonyt(search_term)

def extract_yt_term(command):
    """ Extracts the search term for YouTube from user input. """
    pattern = r'play\s+(.*?)\s+on\s+youtube'
    match = re.search(pattern, command, re.IGNORECASE)
    return match.group(1) if match else ""

def chatBot(query, emotion="neutral"):
    """ Uses HugChat to respond to user queries with emotion-based replies. """
    global chatbot, conversation_id  

    # Modify chatbot response style based on emotion
    if emotion == "happy":
        prompt = query + " Respond cheerfully with positivity in 50 words."
    elif emotion == "sad":
        prompt = query + " Respond in a comforting and empathetic way in 40 words."
    elif emotion == "angry":
        prompt = query + " Respond calmly and logically in 25 words."
    else:
        prompt = query + " Respond naturally in 30 words."

    response = chatbot.chat(prompt)
    print(response)
    speak(response, emotion)  # Pass emotion to speak()
    return response

def extract_features(audio_data):
    """ Extracts MFCC features from speech recognition AudioData object. """
    try:
        # Convert AudioData to raw bytes
        audio_bytes = audio_data.get_wav_data()

        # Convert bytes to NumPy array using soundfile
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Extract MFCC (Mel-Frequency Cepstral Coefficients) features
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)#changed to 13 from 40

        # Compute mean of MFCC features
        mfccs_mean = np.mean(mfccs.T, axis=0)

        return mfccs_mean

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(13)  # Return a default feature vector if extraction fails