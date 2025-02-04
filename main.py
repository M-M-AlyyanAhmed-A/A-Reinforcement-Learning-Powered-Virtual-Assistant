import os
import json
import threading
import speech_recognition as sr

from utils import config
from assistant import core, speech, functions
from rl import environment, agent

def main_loop():
    """The main event loop that listens for commands."""
    R = sr.Recognizer()
    SOURCE = sr.Microphone()
    env = environment.AssistantEnvironment()
    actions = [
            "I'm doing well, thank you for asking!",
            "How are you doing today?",
            "I am ready to help you!",
            "What is your query?"
        ]
    rl_agent = agent.QLearningAgent(actions=actions)
    num_episodes = 5 # Number of learning iterations

    for i in range(num_episodes):
        print(f"Beginning episode {i}")
        state = env.reset() #reset the environment
        done = False
        while not done:
        # User Prompt - we will get this from the microphone eventually
            with sr.Microphone() as source:
                print("Listening for wake word...")
                try:
                    # Adjust timeout and phrase_time_limit for better control
                    audio = R.listen(source, timeout=10, phrase_time_limit=5)
                    # Process on a new thread to not block the main loop
                    threading.Thread(target=core.process_command, args=(audio, env, rl_agent, functions, speech)).start()


                except sr.WaitTimeoutError:
                    print("No speech detected. Retrying...")
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    break
                # The thread is handled above so we continue with the next loop
                continue
        print(f"Episode {i} Complete!")

    print("Training Completed!")
if __name__ == "__main__":
    # Load Environment Variables
    config.load_env()
    main_loop()