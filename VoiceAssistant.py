pip install speechrecognition pyttsx3 pywhatkit wikipedia wolframalpha pyaudio



import speech_recognition as sr
import pyttsx3
import pywhatkit
import wikipedia
import datetime
import wolframalpha
import webbrowser
import os

engine = pyttsx3.init()
engine.setProperty("rate", 160)  
engine.setProperty("volume", 1)  

WOLFRAM_API_KEY = "YOUR_WOLFRAMALPHA_API_KEY" 
wolfram_client = wolframalpha.Client(WOLFRAM_API_KEY)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        
        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand. Please repeat.")
            return None
        except sr.RequestError:
            print("Speech service is unavailable.")
            return None

def execute_command(command):
    if "play" in command:
        song = command.replace("play", "").strip()
        speak(f"Playing {song} on YouTube")
        pywhatkit.playonyt(song)

    elif "time" in command:
        time = datetime.datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {time}")

    elif "search" in command:
        query = command.replace("search", "").strip()
        speak(f"Searching for {query}")
        pywhatkit.search(query)

    elif "wikipedia" in command:
        topic = command.replace("wikipedia", "").strip()
        speak(f"Searching Wikipedia for {topic}")
        result = wikipedia.summary(topic, sentences=2)
        speak(result)

    elif "calculate" in command:
        query = command.replace("calculate", "").strip()
        res = wolfram_client.query(query)
        answer = next(res.results).text
        speak(f"The answer is {answer}")

    elif "open" in command:
        app_name = command.replace("open", "").strip()
        if "chrome" in app_name:
            speak("Opening Google Chrome")
            os.system("start chrome")
        elif "notepad" in app_name:
            speak("Opening Notepad")
            os.system("notepad")
        else:
            speak(f"Sorry, I can't open {app_name} yet.")

    elif "exit" in command or "stop" in command:
        speak("Goodbye!")
        exit()

    else:
        speak("Sorry, I didn't understand that.")

speak("Hello! How can I assist you today?")
while True:
    command = listen()
    if command:
        execute_command(command)
