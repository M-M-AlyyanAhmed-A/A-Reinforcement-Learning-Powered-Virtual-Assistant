import os
from dotenv import load_dotenv

def load_env(dotenv_path = 'F:\Projects\VoiceAssistant\key.env'):
    """Loads environment variables from a .env file"""
    load_dotenv(dotenv_path=dotenv_path)