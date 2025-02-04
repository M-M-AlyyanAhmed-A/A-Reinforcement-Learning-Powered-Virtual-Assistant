class AssistantEnvironment:
    def __init__(self, initial_prompt="Hello, how can I help you?"):
        self.history = [{"role": "user", "content": initial_prompt}]
        self.current_state = self._get_state()

    def _get_state(self):
        # This will be updated to include more complex state information
        return "Current Conversation: " + " ".join(entry['content'] for entry in self.history)

    def step(self, action):
        """
        Args:
            action (str): The assistant's response

        Returns:
            (dict): new state, reward, done
        """
        self.history.append({"role": "assistant", "content": action})
        self.current_state = self._get_state()
        reward = self._calculate_reward(action)
        done = False
        return self.current_state, reward, done

    def _calculate_reward(self, action):
        # Placeholder reward function, this needs refining
        if "I'm doing well" in action:
            return 1  # Good answer
        else:
            return -1  # Bad answer

    def reset(self):
        self.history = [{"role": "user", "content": "Hello, how can I help you?"}]
        self.current_state = self._get_state()
        return self.current_state
