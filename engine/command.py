import pyttsx3
import speech_recognition as sr
import eel
import time
import joblib
import numpy as np



# Load the trained emotion recognition model
emotion_model = joblib.load("c:/users/vpkr1/Desktop/mini_project/engine/emotion_model.pkl")
#if it's a tuple,extract only the model
if isinstance(emotion_model, tuple):
    emotion_model = emotion_model[0]
else:
    emotion_model = emotion_model
    print("DEBUG: Model loaded successfully")

def speak(text, emotion="neutral"):
    """ Speaks the given text with voice adjustments based on detected emotion. """
    text = str(text)
    engine = pyttsx3.init()
    eel.DisplayMessage(text)

    voices = engine.getProperty("voices")  # Get available voices
    engine.setProperty("voice", voices[0].id)  # Default voice

    # Adjust voice tone and speed based on emotion
    if emotion == "happy":
        engine.setProperty("rate", 190)  # Faster speed
        engine.setProperty("volume", 1.0)  # Louder
        engine.setProperty("voice", voices[1].id)  # Change voice
    elif emotion == "sad":
        engine.setProperty("rate", 130)  # Slower speed
        engine.setProperty("volume", 0.7)  # Softer
        engine.setProperty("voice", voices[0].id)  # Default voice
    elif emotion == "angry":
        engine.setProperty("rate", 180)  # Slightly fast
        engine.setProperty("volume", 1.0)  # Normal volume
        engine.setProperty("voice", voices[2].id)  # Deeper voice
    else:
        engine.setProperty("rate", 160)  # Default speed
        engine.setProperty("volume", 0.9)  # Default volume

    engine.say(text)
    eel.receiverText(text)
    engine.runAndWait()

def takecommand():
    """ Listens to user speech, converts it to text, and detects emotion. """
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('Listening...')
        eel.DisplayMessage('Listening...')
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)

        audio = r.listen(source, 10, 6)  # Capture user's voice

    try:
        print('Recognizing...')
        eel.DisplayMessage('Recognizing...')
        query = r.recognize_google(audio, language='en-in')  # Convert speech to text
        print(f'User said: {query}')
        eel.DisplayMessage(query)
        time.sleep(2)

        # Extract features from the recorded audio
        from engine.features import extract_features
        print("debug: Extracting features") # Debugging
        features = extract_features(audio)[:13]#keep only 13 features
        print(f"DEGUG: Extracted Features shape = {features.shape}") #debugging
        #features = np.array(features).reshape(1, -1)  # Reshape for model input

        global emotion_model
        if emotion_model is None:
            print("DEBUG: Model not loaded")
            return query.lower(), "neutral"
        # Predict emotion using the trained model
        try:
            #emotional mapping
            emotion_mapping = {0: "happy", 1: "sad", 2: "angry", 3: "neutral"}

            features = np.array([features],dtype=np.float32).reshape(1, -1) #reshape for model input
            import warnings
            warnings.simplefilter(action='ignore', category=UserWarning)

            predicted_label = emotion_model.predict(features)[0]
            predicted_label=emotion_model.predict(features)[0] #get numerical label
            print(f"DEBUG: Features Shape Before Prediction: {features.shape}")  # Debugging
            print(f"DEBUG: Features Type: {type(features)}")  # Debugging
            emotion = emotion_mapping.get(predicted_label, "neutral") #get the corresponding emotion
            print(f"DEGUG: Detected Emotion = {emotion}")#Debugging
            print(f"Detected Emotion: {emotion}")
        except Exception as e:
            print(f"Error: {e}")
            emotion = "neutral"
        eel.DisplayMessage(f"Emotion: {emotion}")

    except Exception as e:
        return " ", "neutral"  # Default to neutral if there's an error
    print(f"DEBUG: Predicted Emotion = {emotion}")
    return query.lower(), emotion

@eel.expose
def allCommands(message=1):
    """ Processes user commands, detects emotion, and responds accordingly. """
    if message == 1:
        query, emotion = takecommand()  # Get user input and emotion
        print(query)
        eel.senderText(query)
    else:
        query = str(message)
        emotion = "neutral"

    try:
        if "open" in query:
            from engine.features import openCommand
            openCommand(query)
        elif "on youtube" in query:
            from engine.features import PlayYoutube
            PlayYoutube(query)
        else:
            from engine.features import chatBot
            chatBot(query, emotion)  # Pass emotion to chatbot
    except Exception as e:
        print(f"Error: {e}")

    eel.ShowHood()