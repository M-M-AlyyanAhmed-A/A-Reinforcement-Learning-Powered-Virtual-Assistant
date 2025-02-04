import os
import pyttsx3
import threading
import queue
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()
# Environment variables
NUM_CORES = os.cpu_count()
WHISPER_SIZE = 'base'

WHISPER_MODEL = WhisperModel(
    WHISPER_SIZE,
    device='cpu',
    compute_type='int8',
    cpu_threads=NUM_CORES // 2,
    num_workers=NUM_CORES // 2
)

# Initialize a queue for speech requests.
speech_queue = queue.Queue()

# Create thread for processing speech requests.
def speech_worker():
    engine = pyttsx3.init()
    while True:
        try:
            text, event = speech_queue.get()
            if text is None: #If we get none exit the thread.
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"Error in speech_worker: {e}")
            finally:
                event.set()
                speech_queue.task_done() # signal that this task is done so the queue isn't blocked

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error in speech_worker: {e}")

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_non_blocking(text):
    """Function to make the assistant speak without blocking the rest of the program."""
    speaking_event = threading.Event()
    speech_queue.put((text, speaking_event))
    return speaking_event

def wav_to_text(audio_path):
    segments, _ = WHISPER_MODEL.transcribe(audio_path)
    text = " ".join(segment.text for segment in segments)
    return text