import random
import math
import sys
from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here **

        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        ghostDistances = [distance.manhattan(newPos, ghost.getPosition())
                          for ghost in newGhostStates]
        minGhostDist = min(ghostDistances)
        if minGhostDist > 2:
            minGhostDist = 0
        food = currentGameState.getFood()
        foods = food.asList()
        minFoodDist = math.inf
        for food in foods:
            minFoodDist = min(minFoodDist, distance.manhattan(food, newPos))
        danger = (minGhostDist + 0.001)
        profit = (minFoodDist - 100)
        score = successorGameState.getScore() + 1 / profit / danger
        return score
class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.nodes_expanded = 0

    def getAction(self, gameState):
        evalfn = self.getEvaluationFunction()
        num_agents = gameState.getNumAgents()
        tree_depth = self.getTreeDepth()

        def max_value(state, agent_idx, depth):
            if depth == tree_depth or state.isOver():
                return evalfn(state)
            v = -(sys.maxsize - 1)
            if agent_idx == 0:
                actions = state.getLegalActions(agent_idx)
                actions.remove('Stop')
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = max(v, min_value(next_state, agent_idx + 1, depth))
                return v

        def min_value(state, agent_idx, depth):
            v = sys.maxsize
            if depth == tree_depth or state.isOver():
                return evalfn(state)
            if agent_idx == (num_agents - 1):
                actions = state.getLegalActions(agent_idx)
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = min(v, max_value(next_state, 0, depth + 1))
                return v
            else:
                actions = state.getLegalActions(agent_idx)
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = min(v, min_value(next_state, agent_idx + 1, depth))
                return v

        pacman_moves = []
        actions = gameState.getLegalActions()
        actions.remove('Stop')
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            agent_idx = next_state.getLastAgentMoved()
            item = (action, min_value(next_state, agent_idx + 1, 0))
            pacman_moves.append(item)
        action_to_take = max(pacman_moves, key=lambda pacman_moves: pacman_moves[1])
        return action_to_take[0]
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        evalfn = self.getEvaluationFunction()
        num_agents = gameState.getNumAgents()
        tree_depth = self.getTreeDepth()

        def max_value(state, agent_idx, depth, alpha, beta):
            if depth == tree_depth or state.isOver():
                return evalfn(state)
            v = -(sys.maxsize - 1)
            if agent_idx == 0:
                actions = state.getLegalActions(agent_idx)
                actions.remove('Stop')
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = max(v, min_value(next_state, agent_idx + 1, depth, alpha, beta))
                    if v >= beta:
                        return v
                    alpha = max(alpha, v)
                return v

        def min_value(state, agent_idx, depth, alpha, beta):
            v = sys.maxsize
            if depth == tree_depth or state.isOver():
                return evalfn(state)
            if agent_idx == (num_agents - 1):
                actions = state.getLegalActions(agent_idx)
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = min(v, max_value(next_state, 0, depth + 1, alpha, beta))
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
                return v
            else:
                actions = state.getLegalActions(agent_idx)
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = min(v, min_value(next_state, agent_idx + 1, depth, alpha, beta))
                    if v <= alpha:
                        return v
                    beta = min(beta, v)
                return v

        pacman_moves = []
        actions = gameState.getLegalActions()
        actions.remove('Stop')
        ninf = -(sys.maxsize - 1)
        pinf = sys.maxsize
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            agent_idx = next_state.getLastAgentMoved()
            item = (action, min_value(next_state, agent_idx + 1, 0, ninf, pinf))
            pacman_moves.append(item)
        action_to_take = max(pacman_moves, key=lambda pacman_moves: pacman_moves[1])
        return action_to_take[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        evalfn = self.getEvaluationFunction()
        num_agents = gameState.getNumAgents()
        tree_depth = self.getTreeDepth()

        def expectiminimax(state, agent_idx, depth, is_chance):
            v = 0
            if depth == tree_depth or state.isOver():
                return evalfn(state)
            # pacman's turn to play
            if agent_idx == 0 and not is_chance:
                v = -(sys.maxsize - 1)
                actions = state.getLegalActions(agent_idx)
                actions.remove('Stop')
                for action in actions:
                    next_state = state.generateSuccessor(agent_idx, action)
                    v = max(v, expectiminimax(next_state, agent_idx + 1, depth, True))
            # at a chance node to win
            elif is_chance:
                v = 0
                # calculate chance node for final agent
                if agent_idx == (num_agents - 1):
                    actions = state.getLegalActions(agent_idx)
                    for action in actions:
                        next_state = state.generateSuccessor(agent_idx, action)
                        v += (expectiminimax(next_state, 0, depth + 1, False))
                # chance node for adversary agents for the question
                elif agent_idx % num_agents != 0:
                    actions = state.getLegalActions(agent_idx)
                    for action in actions:
                        next_state = state.generateSuccessor(agent_idx, action)
                        v += (expectiminimax(next_state, agent_idx + 1, depth, True))
                v = float(v / len(actions))
            return v

        pacman_moves = []
        actions = gameState.getLegalActions()
        actions.remove('Stop')
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            agent_idx = next_state.getLastAgentMoved()
            item = (action, expectiminimax(next_state, agent_idx + 1, 0, True))
            pacman_moves.append(item)
        action_to_take = max(pacman_moves, key=lambda pacman_moves: pacman_moves[1])
        return action_to_take[0]
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    pac_pos = currentGameState.getAgentPosition(0)
    num_agents = currentGameState.getNumAgents()
    ghost_position = [currentGameState.getAgentPosition(i) for i in range(1, num_agents)]
    capsule_position = currentGameState.getCapsules()
    score = currentGameState.getScore()
    num_food = currentGameState.getNumFood()
    food_list = currentGameState.getFood().asList()
    win = 1 if currentGameState.isWin() else 0
    lose = 1 if currentGameState.isLose() else 0
    over = 1 if currentGameState.isOver() else 0
    food_dist = [(1.0 / (distance.euclidean(pac_pos, food))) for food in food_list]
    ghost_dist = [(1.0 / (distance.euclidean(pac_pos, ghost)))
        for ghost in ghost_position if int(distance.euclidean(pac_pos, ghost)) != 0]
    capsule_dist = [(1.0 / distance.euclidean(pac_pos, capsule))
        for capsule in capsule_position if int(distance.euclidean(pac_pos, capsule) != 0)]
    sorted(food_dist)
    sorted(ghost_dist)
    sorted(capsule_dist)
    if len(capsule_dist) == 0:
        capsule_dist = 0
    else:
        capsule_dist = capsule_dist[0]
    num_brave = 0
    time_scared = 0
    closest_scared = sys.maxsize
    for i in range(1, num_agents):
        ghost_state = currentGameState.getAgentState(i)
        if ghost_state.isBraveGhost():
            num_brave += 1
        else:
            time_scared += ghost_state.getScaredTimer()
            ghost_pos = ghost_state.getPosition()
            if distance.euclidean(pac_pos, ghost_pos) < closest_scared:
                closest_scared = distance.euclidean(pac_pos, ghost_pos)
    closest_scared = 0 if closest_scared == sys.maxsize else (1.0 / (closest_scared))
    if num_brave != (num_agents - 1):
        time_scared = float(time_scared) / (num_agents - 1 - num_brave)

    if over == 0:
        utility = score + (10 * food_dist[0]) + (100 * capsule_dist)
        return utility
    else:
        utility = (1000 * win) - ((1000 * lose) + (10 * num_food)) + score
        return utility
class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
