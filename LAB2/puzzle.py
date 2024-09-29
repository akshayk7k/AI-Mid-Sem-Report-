import numpy as np
import copy
import time
from collections import deque

GOAL_STATE = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 0]])

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class Puzzle8:
    def __init__(self, state):
        self.state = state
        self.blank_pos = tuple(np.argwhere(state == 0)[0])

    def get_possible_moves(self):
        moves = []
        x, y = self.blank_pos
        for dx, dy in DIRECTIONS:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                moves.append((new_x, new_y))
        return moves

    def move(self, new_pos):
        new_x, new_y = new_pos
        x, y = self.blank_pos
        new_state = copy.deepcopy(self.state)
        new_state[x, y], new_state[new_x, new_y] = new_state[new_x, new_y], new_state[x, y]
        return Puzzle8(new_state)

    def is_goal(self):
        return np.array_equal(self.state, GOAL_STATE)

    def __repr__(self):
        return str(self.state)

def uniform_cost_search(initial_state):
    start_time = time.time()
    visited = set()
    queue = deque([(initial_state, [])])

    while queue:
        current_state, path = queue.popleft()
        if current_state.is_goal():
            end_time = time.time()
            print("Time taken:", end_time - start_time)
            return path + [current_state]
        
        visited.add(tuple(map(tuple, current_state.state)))

        for move in current_state.get_possible_moves():
            new_state = current_state.move(move)
            if tuple(map(tuple, new_state.state)) not in visited:
                queue.append((new_state, path + [current_state]))
    
    return None

def iterative_deepening_search(initial_state, max_depth):
    def dls(state, depth, path):
        if state.is_goal():
            return path + [state]
        if depth == 0:
            return None
        
        for move in state.get_possible_moves():
            new_state = state.move(move)
            result = dls(new_state, depth - 1, path + [state])
            if result is not None:
                return result
        return None

    for depth in range(max_depth + 1):
        result = dls(initial_state, depth, [])
        if result is not None:
            return result
    return None

if __name__ == "__main__":
    initial_state = Puzzle8(np.array([[1, 2, 3],
                                       [4, 0, 5],
                                       [7, 8, 6]]))
    
    print("Initial State:")
    print(initial_state)

    print("Solving using Uniform Cost Search...")
    solution = uniform_cost_search(initial_state)
    if solution:
        print("Solution found with Uniform Cost Search:")
        for step in solution:
            print(step)

    print("Solving using Iterative Deepening Search...")
    max_depth = 20
    solution_ids = iterative_deepening_search(initial_state, max_depth)
    if solution_ids:
        print("Solution found with Iterative Deepening Search:")
        for step in solution_ids:
            print(step)
