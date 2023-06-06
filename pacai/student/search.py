from queue import PriorityQueue
from queue import Queue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    frontier = [(problem.startingState(), [])]
    visited = set()
    while frontier:
        state, actions = frontier.pop()
        if problem.isGoal(state):
            return actions
        visited.add(state)
        for successor, action, stepCost in problem.successorStates(state):
            if successor not in visited:
                new_actions = actions + [action]
                frontier.append((successor, new_actions))
    return None

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    start_state = problem.startingState()
    frontier = Queue()
    frontier.put(start_state)
    actions = {start_state: []}
    visited = set()
    while not frontier.empty():
        state = frontier.get()
        if problem.isGoal(state):
            return actions[state]
        visited.add(state)
        for successor, action, stepCost in problem.successorStates(state):
            if successor not in visited:
                frontier.put(successor)
                visited.add(successor)
                actions[successor] = actions[state] + [action]
    return None

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    frontier = PriorityQueue()
    frontier.put((0, problem.startingState(), []))
    current_costs = {problem.startingState(): 0}
    visited = set()
    while not frontier.empty():
        cost, state, actions = frontier.get()
        if problem.isGoal(state):
            return actions
        visited.add(state)
        for successor, action, stepCost in problem.successorStates(state):
            new_actions = actions + [action]
            new_cost = current_costs[state] + stepCost
            if successor not in visited or new_cost < current_costs[successor]:
                current_costs[successor] = new_cost
                frontier.put((new_cost, successor, new_actions))
    return None

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    start_state = problem.startingState()
    frontier = PriorityQueue()
    frontier.put((heuristic(start_state, problem), start_state, []))
    current_costs = {start_state: 0}
    visited = set()
    while not frontier.empty():
        f_n, state, actions = frontier.get()
        if problem.isGoal(state):
            return actions
        visited.add(state)
        for successor, action, stepCost in problem.successorStates(state):
            new_actions = actions + [action]
            new_g_n = current_costs[state] + stepCost
            new_f_n = new_g_n + heuristic(successor, problem)
            if successor not in visited or new_g_n < current_costs[successor]:
                current_costs[successor] = new_g_n
                frontier.put((new_f_n, successor, new_actions))
    return None
