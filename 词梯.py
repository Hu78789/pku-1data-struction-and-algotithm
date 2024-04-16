from collections import deque
import sys
class Vertex:
    def __init__(self,id):
        self.key = id
        self.connectedTo = {}
        self.colors = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.disc = 0
        self.fin = 0
    def add_neighbor(self,nbr,weight):
        self.connectedTo[nbr] = weight
    def get_neighbor(self,nbr,weight=0):
        return self.connectedTo.keys()
    # def __lt__(self,o):
    #     return self.id < o.id

    # def setDiscovery(self, dtime):
    #     self.disc = dtime
    #
    # def setFinish(self, ftime):
    #     self.fin = ftime
    #
    # def getFinish(self):
    #     return self.fin
    #
    # def getDiscovery(self):
    #     return self.disc
class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0
    def add_vertex(self,key):
        self.num_vertices+=1
        new_vertex = Vertex(key)
        self.vertices[key] = new_vertex
        return new_vertex
    def get_vertex(self,n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None
    def __len__(self):
        return self.num_vertices
    def __contains__(self, item):
        return item in self.vertices
    def add_edge(self,f,t,cost):
        # 单向加边
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t],cost)

    def get_vertices(self):
        return list(self.vertices.keys())
    def __iter__(self):
        return iter(self.vertices.values())

def build_graph(filename):
    buckets = {}
    the_graph = Graph()
    with open(filename,'r',encoding='utf-8') as file_in:
        all_words = file_in.readlines()
    for line in all_words:
        word = line.strip()
        for i,_ in enumerate(word):
            bucket = f"{word[:i]}_{word[i+1:]}"
            buckets.setdefault(bucket,set()).add(word)
    for similar_words in buckets.values():
        for word1 in similar_words:
            for word2 in similar_words - {word1}:
                the_graph.add_edge(word1,word2,1)
                the_graph.add_edge(word2,word1,1)
    return the_graph

graph1 = build_graph('')
print(len(graph1))
def bfs(start):
    start.distance = 0
    start.previous = None
    vert_queue = deque()
    vert_queue.append(start)
    while len(vert_queue) > 0:
        current = vert_queue.popleft()
        for neighbor in current:
            if neighbor.color == 'white':
                neighbor.color = 'gray'
                neighbor.distance = current.distance+1
                neighbor.previous = current

                vert_queue.append(neighbor)

        current.color = 'black'
bfs(graph1.get_vertex('FOOL'))
def traverse(starting_vertex):
    ans = []
    current = starting_vertex
    while (current.previous):
        ans.append(current.key)
        current = current.previous
    ans.append(current.key)
    return ans
ans = traverse(graph1.get_vertex('SAGE'))
print(*ans[::-1])




























