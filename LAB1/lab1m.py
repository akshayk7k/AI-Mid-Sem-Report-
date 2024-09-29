from collections import deque
import time

def is_valid(state):
    missionaries, cannibals, boat = state
    if missionaries < 0 or cannibals < 0 or missionaries > 3 or cannibals > 3:
        return False
    if missionaries > 0 and missionaries < cannibals:
        return False
    if (3 - missionaries) > 0 and (3 - missionaries) < (3 - cannibals):
        return False
    return True

def get_successors(state):
    successors = []
    missionaries, cannibals, boat = state
    moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
    
    if boat == 1:
        for move in moves:
            new_state = (missionaries - move[0], cannibals - move[1], 0)
            if is_valid(new_state):
                successors.append(new_state)
    else:
        for move in moves:
            new_state = (missionaries + move[0], cannibals + move[1], 1)
            if is_valid(new_state):
                successors.append(new_state)
    return successors

def bfs(start_state, goal_state):
    queue = deque([(start_state, [])])
    visited = set()
    while queue:
        (state, path) = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            return path
        for successor in get_successors(state):
            queue.append((successor, path))
    return None

def dfs(start_state, goal_state):
    stack = [(start_state, [])]
    visited = set()
    while stack:
        (state, path) = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        path = path + [state]
        if state == goal_state:
            return path
        for successor in get_successors(state):
            stack.append((successor, path))
    return None

def print_solution(solution):
    if solution:
        print("Solution found in", len(solution) - 1, "steps:")
        for step in solution:
            print(step)
    else:
        print("No solution found.")

def compare_search_algorithms():
    start_state = (3, 3, 1)
    goal_state = (0, 0, 0)
    
    start_time = time.perf_counter()
    bfs_solution = bfs(start_state, goal_state)
    bfs_time = time.perf_counter() - start_time
    print("\nBFS Solution:")
    print_solution(bfs_solution)
    print(f"BFS Time Complexity: {bfs_time:.8f} seconds")
    
    start_time = time.perf_counter()
    dfs_solution = dfs(start_state, goal_state)
    dfs_time = time.perf_counter() - start_time
    print("\nDFS Solution:")
    print_solution(dfs_solution)
    print(f"DFS Time Complexity: {dfs_time:.8f} seconds")
    
    if bfs_time < dfs_time:
        print("\nBFS was faster.")
    else:
        print("\nDFS was faster.")

compare_search_algorithms()
