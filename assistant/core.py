import time
import speech_recognition as sr
from utils import memory
from assistant import speech
from dotenv import load_dotenv
import os
from groq import Groq
import google.generativeai as genai

load_dotenv('F:\Projects\VoiceAssistant\key.env')

# Environment variables
WAKE_WORD = os.getenv("WAKE_WORD", "alex")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
# Initialize conversation memory
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVERSATION_MEMORY = os.path.join(BASE_PATH, "conversation_memory.json")
if not os.path.exists(CONVERSATION_MEMORY):
    with open(CONVERSATION_MEMORY, 'w') as f:
        json.dump([], f)

groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GOOGLE_AI_API_KEY)

SYS_MSG = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

CONVO = [{'role': 'system', 'content': SYS_MSG}]  # Conversation memory

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

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\nIMAGE CONTEXT: {img_context}'
    CONVO.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=CONVO, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    CONVO.append(response)

    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]

    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content


def process_command(audio_data, env, rl_agent, functions, speech):
    """Processes the audio data, handles all command logic and responds"""
    try:
        print("Audio received. Processing...")
        BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        PROMPT_AUDIO_PATH = os.path.join(BASE_PATH, 'prompt.wav')
        # Save the audio to file for debugging and transcription
        with open(PROMPT_AUDIO_PATH, 'wb') as f:
            f.write(audio_data.get_wav_data())

        # Transcribe the audio to text
        prompt_text = speech.wav_to_text(PROMPT_AUDIO_PATH)
        print(f"Transcribed Text: '{prompt_text}'")  # Debugging statement

        # Check if the transcription is valid
        if not prompt_text or len(prompt_text.strip()) == 0:
            print("No valid transcription detected.")
            return

        # Trim extra spaces from the transcription
        prompt_text = prompt_text.strip()


        # Check if the transcription starts with the wake word
        if prompt_text.lower().startswith(WAKE_WORD.lower()):
             clean_prompt = prompt_text[len(WAKE_WORD):].strip()  # Remove the wake word and extra spaces
             print(f"USER Command: {clean_prompt}")
             # Stop the assistant temporarily if necessary (e.g., if "stop" or "husky" is mentioned)
             if 'husky' in clean_prompt or 'stop' in clean_prompt:
                 print("Assistant paused. Waiting for 5 seconds...")
                 time.sleep(5)  # Pause before continuing
                 return  # Exit early to prevent further processing

             # Proceed with handling the command
             call = function_call(clean_prompt)

             # Handle visual context (taking screenshots, capturing webcam, etc.)
             visual_context = None
             if 'take screenshot' in call:
                 print('Action: Taking Screenshot.')
                 functions.take_screenshot()
                 visual_context = functions.vision_prompt(prompt=clean_prompt, photo_path=functions.SCREENSHOT_PATH)

             elif 'capture webcam' in call:
                 print('Action: Capturing Webcam Image.')
                 camera_path = functions.web_cam_capture()
                 if camera_path:
                     visual_context = functions.vision_prompt(prompt=clean_prompt, photo_path=camera_path)

             elif 'extract clipboard' in call:
                 print('Action: Extracting Clipboard Content.')
                 paste = functions.get_clipboard_text()
                 if paste:
                     clean_prompt = f'{clean_prompt}\n\nCLIPBOARD CONTENT: {paste}'

             # Generate the response for other cases
             response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
             print(f"ASSISTANT Response: {response}")

             #Get the assistants response
             state = env._get_state()
             action = rl_agent.choose_action(state)
             print(f"Assistant: {action}")
             next_state, reward, done = env.step(action)
             print(f"Reward: {reward}")

              # Learn
             rl_agent.learn(state, action, reward, next_state, done)

             event = speech.speak_non_blocking(response)
             event.wait()
             memory.save_conversation(clean_prompt, response)

        else:
           print(f"No valid wake word detected in: '{prompt_text}'")


    except sr.UnknownValueError:
        print("Speech Recognition: Could not understand the audio.")
    except sr.RequestError as e:
        print(f"Speech Recognition Service Error: {e}")
    except Exception as e:
        print(f"Error in process_command: {e}")
    finally:
        print("Ready for the next command...")