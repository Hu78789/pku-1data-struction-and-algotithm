from collections import deque
class Graph:
    def __init__(self,grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.visited = set()
    def is_valid_move(self,row,col):
        return (0<=row<self.rows and 0<=col<self.cols
                and self.grid[row][col] == 'W' and (row,col) not in self.visited)
    def bfs(self,start_row,start_col):
        queue = deque([(start_row,start_col)])
        self.visited.add((start_row,start_col))
        area = 1
        while queue:
            r,c = queue.popleft()
            for dr,dc in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
                new_row,new_col = r+dr,c+dc
                if self.is_valid_move(new_row,new_col):
                    queue.append((new_row,new_col))
                    self.visited.add((new_row,new_col))
                    area+=1
        return area
def max_connected_area(grid):
    graph = Graph(grid)
    max_area = 0
    for i in range(graph.rows):
        for j in range(graph.cols):
            if grid[i][j] == 'W' and (i,j) not in graph.visited:
                area = graph.bfs(i,j)
                max_area = max(area,max_area)
    return max_area
T = int(input())
for _ in range(T):
    N,M = map(int,input().split())
    grid = [input() for _ in range(N)]
    result = max_connected_area(grid)
    print(result)






