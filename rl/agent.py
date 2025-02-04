import random

class QLearningAgent:
   def __init__(self, actions):
      self.q_table = {} # Q-value for each state and action
      self.actions = actions
      self.learning_rate = 0.1
      self.discount_factor = 0.9

   def choose_action(self, state):
      if state not in self.q_table:
         self.q_table[state] = {action: 0 for action in self.actions}

      #Check if any actions are equal, otherwise choose max
      max_q = max(self.q_table[state].values())
      max_actions = [action for action, q in self.q_table[state].items() if q == max_q]
      if len(max_actions) == len(self.actions):
        return random.choice(self.actions) # If no better action
      else:
         return max_actions[0] # Choose the action with max q value

   def learn(self, state, action, reward, next_state, done):
       if next_state not in self.q_table:
          self.q_table[next_state] = {action: 0 for action in self.actions}

       if state not in self.q_table:
          self.q_table[state] = {action: 0 for action in self.actions}

       max_next_q = max(self.q_table[next_state].values())
       self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + self.learning_rate * (reward + self.discount_factor * max_next_q)