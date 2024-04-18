import sys
from collections import deque
class Vertex:
    def __init__(self,key):
        self.connectedTo={}
        self.key = key
        self.previous = None
        self.distance = sys.maxsize
        self.color = 'white'
    def add_neighbor(self,nbr,weight = 1):
        #nbr应为顶点
        self.connectedTo[nbr] = weight
    def get_neighbors(self):
        return list(self.connectedTo.keys())

class Graph:
    def __init__(self,id = None):
        self.id = id
        self.vertices = {}
        self.num_vertices = 0
    def add_vertex(self,key):
        self.num_vertices += 1
        new = Vertex(key)
        self.vertices[key] = new
        return new
    def __contains__(self, item):
        return item in self.vertices

    def __len__(self):
        return self.num_vertices
    def get_vertex(self,key):
        return self.vertices[key] if key in self.vertices else None
    def add_edge(self,v1,v2,cost=1):
        #v1->v2
        if v1 not in self.vertices:
            self.add_vertex(v1)
        if v2 not in self.vertices:
            self.add_vertex(v2)
        self.vertices[v1].add_neighbor(self.vertices[v2],cost)

    def get_vertices(self):
        return list(self.vertices.keys())

    def __iter__(self):
        return iter(self.vertices.values())
def build_word_graph(words):
    buckets = {}
    graph = Graph()
    for word in words:
        for i,_ in enumerate(word):
            bucket = f'{word[:i]}_{word[i+1:]}'
            buckets.setdefault(bucket,set()).add(word)
    for similar_words in buckets.values():
        for word1 in similar_words:
            for word2 in similar_words-{word1}:
                graph.add_edge(word1,word2)
    return graph
def bfs(start):
    start.distance = 0
    start.previous = None
    vert_queue = deque()
    vert_queue.append(start)
    while len(vert_queue):
        current = vert_queue.popleft()
        for neighbor in current.get_neighbors():

            if neighbor.color == 'white':
                neighbor.color = 'gray'
                neighbor.distance = current.distance + 1
                neighbor.previous = current
                vert_queue.append(neighbor)
        current.color = 'black'
        #print(current)
def traverse_path(words,start,end):
    graph = build_word_graph(words)

    bfs(graph.get_vertex(start))
    ans = []
    if end not in graph:
        return None
    current = graph.get_vertex(end)
    while (current.previous):
        #print(current)
        ans.append(current.key)
        current = current.previous
    ans.append(current.key)
    if ans[-1] != start:
        return None
    else:
        return ans[::-1]


n = int(input())
words = []
for _ in range(n):
    words.append(input().strip())

start,end = input().split()


result = traverse_path(words,start,end)
if result:
    print(' '.join(result))
else:
    print('NO')





























