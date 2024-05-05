import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService
import nltk
import config
import os
import sys
import util
import speech_recognition as sr
import pyttsx3

console = Console()

if config.DISABLE_STDOUT:
    null_fd = open(os.devnull, 'w')
    null_fileno = null_fd.fileno()
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    sys.stdout = null_fd
    sys.stderr = null_fd


stt = whisper.load_model(config.WHISPER_MODEL)
tts = TextToSpeechService()
recognizer = sr.Recognizer()

sd.default.device = config.SOUNDDRIVE_SPEAKER_DEVICE_ID

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
voices = engine.getProperty('voices')
engine.setProperty("voice", 'zh')

try:
    nltk.data.find('tokenizers/punkt')
    print("The 'punkt' tokenizer is already downloaded.")
except LookupError:
    print("The 'punkt' tokenizer is not found, downloading now...")
    nltk.download('punkt')
    print("Download complete.")

PROMPT = PromptTemplate(input_variables=["history", "input"], template=config.TEMPLATE)
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(model=config.OLLAMA_MODEL),
)


def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))
        # text = recognizer.recognize_google(np.frombuffer(indata, dtype=np.int16), language="en-US")
        # print("You said: ", text)

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        start_time = time.time()
        while not stop_event.is_set():
            # if data_queue.qsize() > 5:
            #     audio_data = b"".join(list(data_queue.queue))
            #     audio_data = np.frombuffer(audio_data, dtype=np.int16)
            #     # audio_np = (
            #     #     np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            #     # )
            #     text = recognizer.recognize_google(audio_data, language="en-US")
            #     print("You said: ", text)
            time.sleep(0.1)
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > 60:
                data_queue.queue.clear()




def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    response = chain.predict(input=text)
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    return response


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            # voice activation
            while True:
                audio_data = b"".join(list(data_queue.queue))
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                )
                if audio_np.size > 0:
                    text = transcribe(audio_np)
                    if "Hello" in str(text):
                        console.print("Detected: Voice Activated")
                        engine.say(config.VOICE_BEGIN_TEXT)
                        engine.runAndWait()
                        data_queue.queue.clear()
                        break
                time.sleep(1)
            
            console.print("Listening:...")
            time.sleep(10)


            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                    # text2 = recognizer.recognize_google(audio_np, language="en-US")
                console.print(f"[yellow]You: {text}")
                # console.print(f"[yellow]You: {text2}")

                with console.status("Generating response...", spinner="earth"):
                    response = util.limit_words(get_llm_response(text))
                    console.print(f"[cyan]Assistant: {response}")
                
                with console.status("Generating voice...", spinner="dots"):
                    sample_rate, audio_array = tts.long_form_synthesize(response)

                play_audio(sample_rate, audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
    if config.DISABLE_STDOUT:
        null_fd.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

   
