# value_iteration_agents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

#comentario para que se vuelva a subir


# value_iteration_agents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learning_agents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state, action, next_state)
              mdp.is_terminal(state)
        """
        super().__init__()
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.run_value_iteration()
    
    def run_value_iteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        # Get the list of all states in the MDP
        states = self.mdp.get_states()

        # Iterate for a specified number of iterations (or until convergence, depending on the setup)

        for i in range(self.iterations):
            new_values = util.Counter() # Create a counter to hold the new values for this iteration
            # Iterate over each state in the MDP

            for state in states:
                if self.mdp.is_terminal(state):
                    continue
                # Initialize the maximum value for this state as negative infinity
                max_value = float('-inf')
                # For each action possible in the state, initialize the Q-value for this action to 0

                for action in self.mdp.get_possible_actions(state):
                    q_value = 0
                    # For each possible next state and probability resulting from this action, get the reward for transition and update the Q-value

                    for next_state, prob in self.mdp.get_transition_states_and_probs(state, action):
                        reward = self.mdp.get_reward(state, action, next_state)
                        q_value += prob * (reward + self.discount * self.values[next_state])
                    # Update the maximum value for this state if the Q-value is greater than the current maximum value
                    max_value = max(max_value, q_value)
                # Update the new values for this iteration with the maximum value for this state
                new_values[state] = max_value
            # Update the values for the next iteration with the new values
            self.values = new_values
        


            
    def get_value(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def compute_q_value_from_values(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #Initialize total Q-value as 0
        total = 0
        #Get the possible next states and probabilities resulting from the action in the
        transStatesAndProbs = self.mdp.get_transition_states_and_probs(state, action)

        
        #For each possible next state and probability, get the reward for the transition and update the Q-value
        for tranStateAndProb in transStatesAndProbs:
            tstate = tranStateAndProb[0]
            prob = tranStateAndProb[1]
            reward = self.mdp.get_reward(state, action, tstate)
            value = self.get_value(tstate)
            total += prob * (reward + self.discount * value)
        #Return the total Q-value
        return total


        util.raise_not_defined()


    def compute_action_from_values(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #If the state is terminal, return None
        if self.mdp.is_terminal(state):
            return None
        #Get the possible actions in the state
        actions = self.mdp.get_possible_actions(state)
        if len(actions) == 0:
            return None #redundant because of the above if statement but here just in case
        #Initialize the maximum value for the state as the Q-value of the first action
        max_value = self.get_q_value(state, actions[0])
        max_action = actions[0]
        #For each action in the state, get the Q-value and update the maximum value and action if the Q-value is greater
        for action in actions:
            value = self.get_q_value(state, action)
            if value > max_value:
                max_value = value
                max_action = action
        #Return the action with the maximum Q-value
        return max_action
        util.raise_not_defined()

    def get_policy(self, state):
        return self.compute_action_from_values(state)

    def get_action(self, state):
        """Returns the policy at the state (no exploration)."""
        return self.compute_action_from_values(state)

    def get_q_value(self, state, action):
        return self.compute_q_value_from_values(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):

    """
        * Please read learning_agents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.get_states()
              mdp.get_possible_actions(state)
              mdp.get_transition_states_and_probs(state, action)
              mdp.get_reward(state)
              mdp.is_terminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        """*** YOUR CODE HERE ***"""

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learning_agents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def run_value_iteration(self):
        """*** YOUR CODE HERE ***"""

