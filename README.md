# A Reinforcement Learning Virtual Assistant Project

Alex is an experimental virtual assistant built in Python that leverages reinforcement learning to enhance its ability to understand and respond to user requests effectively. This project aims to demonstrate how a virtual assistant, similar to Siri or Alexa, can adapt and improve its performance through interaction and learning using RL.

**Core Technologies:**

*   **Speech Recognition (`speech_recognition`):** Enables the assistant to capture and transcribe user voice commands.
*   **Language Model Integration (Groq and Google Gemini 1.5 Flash):** Employs powerful language models from Groq's Llama3 and Google's Gemini 1.5 Flash to process natural language, understand user intent, and generate relevant responses.
*   **Reinforcement Learning (Q-learning):** Integrates a Q-learning agent to learn optimal responses based on a defined reward function, continuously improving the assistant's ability to satisfy user requests and maintain coherent conversations.
*   **Speech Synthesis (`pyttsx3`):** Provides text-to-speech capabilities, allowing the assistant to communicate its responses audibly.

**Key Features:**

*   **Wake Word Detection:** Supports multiple wake words (e.g., "alex", "hey alex") for more natural interaction.
*   **Interruptible Speech:** Users can interrupt the assistant's speech with a specific command (e.g., "alex, stop").
*   **Shutdown Command:** Includes a command (e.g., "alex, shut down") for safely terminating the program.
*   **Conversation Memory:** Stores the conversation history to provide context for more relevant responses.
*   **Function Calling:** The AI can extract clipboard data, take screenshots, and capture webcam images.

**Project Structure:**

The project follows a modular structure:

*   `assistant/`: Contains core assistant functionality (speech processing, LLM interaction, function calls).
*   `rl/`: Implements the reinforcement learning components (agent, environment).
*   `utils/`: Provides utility modules (configuration, conversation memory).

**Current Status and Roadmap:**

This is a work-in-progress project that explores the integration of RL with virtual assistants.  Future development efforts will focus on:

*   Improving the RL implementation, including more sophisticated state representation and reward functions.
*   Adding more advanced natural language understanding capabilities.
*   Expanding the set of available functions and actions.
*   Enhancing the overall user experience.

## Usage

1.  **Clone the Repository:**

    ```bash
    git clone [your repository URL]
    cd [repository directory]
    ```

2.  **Set up the Environment:**
   Install python 3.9 or greater.
   Create a virtual envirnoment using the command `python -m venv .venv`.
   Activate the virtual envirnment by running `.venv\Scripts\activate`.

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**

    *   Create a `.env` file in the project root directory.
    *   Add the following environment variables, replacing the placeholders with your actual API keys and values:

        ```
        WAKE_WORD="alex"
        GROQ_API_KEY="YOUR_GROQ_API_KEY"
        GOOGLE_AI_API_KEY="YOUR_GOOGLE_AI_API_KEY"
        GOOGLE_SEARCH_ENGINE_ID="YOUR_GOOGLE_SEARCH_ENGINE_ID" (Optional)
        ```

5.  **Run the Assistant:**

    ```bash
    python main.py
    ```

6.  **Interact with the Assistant:**

    *   Wait for the assistant to indicate that it's listening for a wake word.
    *   Speak the wake word (e.g., "alex") followed by your command or question.
    *   To interrupt the assistant's speech, say the wake word followed by "stop".
    *   To shut down the assistant, say the wake word followed by "shut down" or "good bye".

**Important Notes:**

*   Ensure your microphone is properly configured.
*   The assistant requires internet access to connect to the Groq and Google Gemini APIs.
*   The quality of the speech recognition and responses depends on the accuracy of the ASR and LLM services.
*   This is a development project, and results may vary.
