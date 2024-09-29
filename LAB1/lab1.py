from collections import deque

initial_state = ('E', 'E', 'E', '_', 'W', 'W', 'W')
goal_state = ('W', 'W', 'W', '_', 'E', 'E', 'E')

def get_neighbors(state):
    neighbors = []
    index_empty = state.index('_')
    
    if index_empty > 0:
        new_state = list(state)
        new_state[index_empty], new_state[index_empty - 1] = new_state[index_empty - 1], new_state[index_empty]
        neighbors.append(tuple(new_state))
    
    if index_empty < len(state) - 1:
        new_state = list(state)
        new_state[index_empty], new_state[index_empty + 1] = new_state[index_empty + 1], new_state[index_empty]
        neighbors.append(tuple(new_state))
    
    if index_empty > 1:
        new_state = list(state)
        new_state[index_empty], new_state[index_empty - 2] = new_state[index_empty - 2], new_state[index_empty]
        neighbors.append(tuple(new_state))
    
    if index_empty < len(state) - 2:
        new_state = list(state)
        new_state[index_empty], new_state[index_empty + 2] = new_state[index_empty + 2], new_state[index_empty]
        neighbors.append(tuple(new_state))
    
    return neighbors

def bfs(initial_state, goal_state):
    queue = deque([(initial_state, [])])
    visited = set()
    
    while queue:
        state, path = queue.popleft()
        
        if state in visited:
            continue
        
        visited.add(state)
        
        if state == goal_state:
            return path + [state]
        
        for neighbor in get_neighbors(state):
            queue.append((neighbor, path + [state]))

def dfs(initial_state, goal_state):
    stack = [(initial_state, [])]
    visited = set()
    
    while stack:
        state, path = stack.pop()
        
        if state in visited:
            continue
        
        visited.add(state)
        
        if state == goal_state:
            return path + [state]
        
        for neighbor in get_neighbors(state):
            stack.append((neighbor, path + [state]))

def print_solution(path):
    print("Solution steps:")
    for step in path:
        print(step)

def main():
    print("Solving the problem using BFS...")
    bfs_solution = bfs(initial_state, goal_state)
    print("BFS found a solution in", len(bfs_solution) - 1, "steps:")
    print_solution(bfs_solution)
    
    print("\nSolving the problem using DFS...")
    dfs_solution = dfs(initial_state, goal_state)
    print("DFS found a solution in", len(dfs_solution) - 1, "steps:")
    print_solution(dfs_solution)
    
    print("\nComparison of BFS and DFS:")
    print("BFS Steps:", len(bfs_solution) - 1)
    print("DFS Steps:", len(dfs_solution) - 1)
    
    print("\nTime and Space Complexity:")
    print("BFS Time Complexity: O(b^d), where b is the branching factor and d is the depth of the solution.")
    print("DFS Time Complexity: O(b^m), where m is the maximum depth of the search.")
    print("BFS Space Complexity: O(b^d) because BFS stores all nodes at each level.")
    print("DFS Space Complexity: O(b * m) because DFS stores only the current path.")

if __name__ == "__main__":
    main()
