from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection, probability
import random
import sys

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.qvalues = {}
        self.initializeqvalues()

    def initializeqvalues(self):
        self.qvalues = {}

    def getAction(self, state):
        cond = probability.flipCoin(1 - self.getEpsilon())
        actions = self.getLegalActions(state)
        if cond:
            return self.getPolicy(state)
        else:
            ret = random.choice(actions)
        return ret

    def update(self, state, action, nextState, reward):
        discount = self.getDiscountRate()
        alpha = self.getAlpha()
        best = self.getValue(nextState)
        sample = reward + (discount * best)
        new_qval = ((1 - alpha) * self.getQValue(state, action)) + (alpha * sample)
        self.qvalues[(state, action)] = new_qval

    def getQValue(self, state, action):
        item = (state, action)
        if item not in self.qvalues:
            return 0.0
        return self.qvalues[(state, action)]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return 0.0
        best = -(sys.maxsize - 1)
        for action in actions:
            q_val = self.getQValue(state, action)
            if q_val > best:
                best = q_val

        return best

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        best = -(sys.maxsize - 1)
        best_actions = []
        for action in actions:
            q_val = self.getQValue(state, action)
            if q_val > best:
                best = q_val
                best_actions = [action]
            elif q_val == best:
                best_actions.append(action)
        return random.choice(best_actions)

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)
        self.weights = {}

    def initializeWeights(self):
        """
        Initialize the weights dictionary.
        """
        # You can customize the initialization of weights based on your feature extractor.
        # For example, you can set all weights to 0.0 initially or use random values.
        self.weights = {}

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            pass

    def getQValue(self, state, action):
        q_val = 0.0
        features = self.featExtractor.getFeatures(self, state, action)
        for k in features:
            weight = self.weights.get(k, 0.0)
            feature = features[k]
            q_val += weight * feature
        return q_val

    def update(self, state, action, nextState, reward):
        discount = self.getDiscountRate()
        alpha = self.getAlpha()
        value = self.getValue(nextState)
        q_val = self.getQValue(state, action)
        features = self.featExtractor.getFeatures(self, state, action)
        correction = (reward + (discount * value) - q_val)
        for elem in features:
            self.weights[elem] = self.weights.get(elem, 0.0) + (alpha * correction) * features[elem]
