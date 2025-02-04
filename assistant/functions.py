from PIL import ImageGrab, Image
import cv2
import pyperclip
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Environment variables
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
# Initialize paths
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_PATH = os.path.join(BASE_PATH, "screenshot.jpg")
WEBCAM_PATH = os.path.join(BASE_PATH, "webcam.jpg")


genai.configure(api_key=GOOGLE_AI_API_KEY)

GENERATION_CONFIG = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

SAFETY_SETTINGS = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
]

MODEL = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config=GENERATION_CONFIG, safety_settings=SAFETY_SETTINGS)

def take_screenshot():
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(SCREENSHOT_PATH, quality=15)

def web_cam_capture():
    web_cam = cv2.VideoCapture(0)  # Use 0 for the default webcam
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        return None  # Return None to signify failure
    ret, frame = web_cam.read()
    if ret:
        cv2.imwrite(WEBCAM_PATH, frame)
        print(f'Webcam image saved as {WEBCAM_PATH}')
        web_cam.release()
        return WEBCAM_PATH
    else:
        print('Failed to capture webcam image')
        web_cam.release()
        return None # Return None to signify failure

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None

def vision_prompt(prompt, photo_path):
    try:
        img = Image.open(photo_path)
    except FileNotFoundError:
        return "Error: Image file not found."
    except Exception as e:
        return f"Error opening image: {e}"

    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )

    try:
        response = MODEL.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"