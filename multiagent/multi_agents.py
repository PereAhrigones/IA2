# multi_agents.py
# --------------
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


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"
            # Initialize evaluation components
        min_food_distance = float("inf")
        ghost_penalty = 0
        scared_ghost_bonus = 0

        # Compute minimum distance to food
        for food in new_food.as_list():
            min_food_distance = min(min_food_distance, manhattan_distance(new_pos, food))

        # Evaluate ghost positions
        for i, ghost in enumerate(successor_game_state.get_ghost_positions()):
            ghost_distance = manhattan_distance(new_pos, ghost)
            if new_scared_times[i] > 0:  # Ghost is scared
                scared_ghost_bonus += 200 / (ghost_distance + 1)  # Reward getting closer to scared ghosts
            elif ghost_distance < 2:  # Ghost is too close and not scared
                ghost_penalty = -float('inf')  # Immediate bad state

        # Penalize staying in the same position (likely a no-op move)
        if new_pos == current_game_state.get_pacman_position():
            return -float('inf')
        
        # Penalize repetitive back-and-forth moves
        if successor_game_state.get_pacman_position() == current_game_state.get_pacman_position():
            ghost_penalty -= 10  # Small penalty to discourage repetitive moves

        # Reward progress toward food or scared ghosts
        progress_reward = 1.0 / (min_food_distance + 1)  # Encourage moving closer to food
        if min_food_distance == float("inf"):  # No food left
            progress_reward = 0

        # Calculate the final score
        score = (
            successor_game_state.get_score()  # Base game score
            + progress_reward  # Reward for moving closer to food
            + scared_ghost_bonus  # Bonus for chasing scared ghosts
            + ghost_penalty  # Penalty for being close to non-scared ghosts
        )

        return score
def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(game_state, 0, 0)[0]
    
    def max_value(self, game_state, depth, agent_index):
        if depth == self.depth or game_state.is_win() or game_state.is_lose():
            return None, self.evaluation_function(game_state)
        max_value = -float("inf")
        max_action = None
        for action in game_state.get_legal_actions(agent_index):
            successor = game_state.generate_successor(agent_index, action)
            _, value = self.min_value(successor, depth, agent_index + 1)
            if value > max_value:
                max_value = value
                max_action = action
        return max_action, max_value
    
    def min_value(self, game_state, depth, agent_index):
        if depth == self.depth or game_state.is_win() or game_state.is_lose():
            return None, self.evaluation_function(game_state)
        min_value = float("inf")
        min_action = None
        for action in game_state.get_legal_actions(agent_index):
            successor = game_state.generate_successor(agent_index, action)
            if agent_index == game_state.get_num_agents() - 1:
                _, value = self.max_value(successor, depth + 1, 0)
            else:
                _, value = self.min_value(successor, depth, agent_index + 1)
            if value < min_value:
                min_value = value
                min_action = action
        return min_action, min_value
    
    

    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    An agent that uses Alpha-Beta pruning to choose the best action.
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function.
        This function initiates the alpha-beta pruning process.
        """
        # Start the alpha-beta pruning process with initial alpha and beta values
        return self.max_value(game_state, 0, -float("inf"), float("inf"))[0]

        util.raise_not_defined()
    
    def max_value(self, game_state, depth, alpha, beta):
        """
        Returns the maximum value for the current game state.
        This function is called for the maximizing player (Pacman).
        """
        # Check if the search has reached the maximum depth or if the game is over
        if depth == self.depth or game_state.is_win() or game_state.is_lose():
            return None, self.evaluation_function(game_state)
        
        max_value = -float("inf")
        max_action = None

        # Iterate over all legal actions for the maximizing player
        for action in game_state.get_legal_actions(0):
            successor = game_state.generate_successor(0, action)
            # Call min_value for the minimizing player (ghosts)
            _, value = self.min_value(successor, depth, 1, alpha, beta)
            if value > max_value:
                max_value = value
                max_action = action
            # Alpha-Beta pruning so
            if max_value > beta:
                return max_action, max_value
            alpha = max(alpha, max_value)
        
        return max_action, max_value
    
    def min_value(self, game_state, depth, agent, alpha, beta):
        if depth == self.depth or game_state.is_win() or game_state.is_lose():
            return None, self.evaluation_function(game_state)
        min_value = float("inf")
        min_action = None
        for action in game_state.get_legal_actions(agent):
            successor = game_state.generate_successor(agent, action)
            if agent == game_state.get_num_agents() - 1:
                _, value = self.max_value(successor, depth + 1, alpha, beta)
            else:
                _, value = self.min_value(successor, depth, agent + 1, alpha, beta)
            if value < min_value:
                min_value = value
                min_action = action
            if min_value < alpha:
                return min_action, min_value
            beta = min(beta, min_value)
        return min_action, min_value


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
