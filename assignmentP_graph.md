# Assignment #P: 课程大作业

Updated 1009 GMT+8 Feb 28, 2024

2024 spring, Complied by ==胡景博 药学==



**说明：**

关乎每位同学维护自己的 GitHub 作业，本意是让大家练习常用于计算机科学学生的代码和文档维护方法。通过计算概论、数据结构和算法等课程，我们希望引导大家进入计算机学科领域。这将帮助同学们熟悉实际的编码和文档管理流程，并培养在团队协作和版本控制方面的技能。

1）提交内容，请填写到下面作业模版中。

2）截止时间是期末出分前，因为Canvas可以多次提交，建议期末机考前提交一次，考试后加上课程总结再提交一次。



评分标准

| 标准           | 等级                                   | 得分       |
| -------------- | -------------------------------------- | ---------- |
| 按时提交       | 1 得分提交，0.5 得分请假，0 得分未提交 | 1 分       |
| 你的GitHub网址 | 1 得分有，0 得分无                     | 1 分       |
| 你的GitHub截图 | 1 得分有，0 得分无                     | 1 分       |
| Cheatsheet     | 1 得分有，0 得分无                     | 1 分       |
| 课程资料和情报 | 1 得分有，0 得分无                     | 1 分       |
| 总得分：       |                                        | 5 ，满分 5 |





## 1. 要求

同学开自己的GitHub，自己数算的学习方法、做的题目、考试时候要带的记录纸（cheat_sheet）等放在上面。方便大家关注，当你有新的更新时，我们也可以及时获得最新的内容。

例子1：https://github.com/forxhunter/libpku 这样的项目可以作为一个数算课程的项目，同时也是同学们整理资料的一个好方式，可以实现一举多得的效果。



![image-20240219114316139](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240219114316139.png)





例子2: https://github.com/PKUanonym/REKCARC-TSC-UHT

![image-20240219114436829](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240219114436829.png)



## 2. 提交内容

你的GitHub网址及截图。
网址：https://github.com/Hu78789/pku-1data-struction-and-algotithm
资源推荐：算法竞赛（罗勇军，郭卫斌）
![alt text](image-27.png)

#cheetpaper
##图算法模块
02488：骑士之旅
查看提交统计提问
总时间限制： 1000ms 内存限制： 65536kB
描述
背景
骑士厌倦了一次又一次地看到相同的黑白方块，并决定
环游世界。每当骑士移动时，它都是一个方向上的两个方格和一个垂直于此方向的方格。骑士的世界就是他赖以生存的棋盘。我们的骑士生活在一个象棋盘上，棋盘的面积比普通的 8 * 8 棋盘小，但它仍然是矩形的。你能帮助这位喜欢冒险的骑士制定旅行计划吗？

问题：
找到一条路径，让骑士访问每个方格一次。骑士可以在棋盘的任何方格上开始和结束。
输入
输入从第一行的正整数 n 开始。以下行包含 n 个测试用例。每个测试用例由一条带有两个正整数 p 和 q 的行组成，使得 1 <= p * q <= 26。这表示一个 p * q 棋盘，其中 p 描述存在多少个不同的平方数 1， . . . ， p，q 描述存在多少个不同的平方字母。这些是拉丁字母表的前 q 字母：A、. . .
输出
每个方案的输出都以包含“方案 #i：” 的行开头，其中 i 是从 1 开始的方案编号。然后打印一行，其中包含字典上的第一条路径，该路径以骑士步法访问棋盘的所有方格，然后是空行。路径应通过连接访问的方块的名称在一行上给出。每个方块名称由一个大写字母后跟一个数字组成。
如果不存在这样的路径，则应在单行上输出不可能。
样例输入
3
1 1
2 3
4 3
样例输出
Scenario #1:
A1

Scenario #2:
impossible

Scenario #3:
A1B3C1A2B4C2A3B1C3A4B2C4

```python
move = [(-2,-1),(-2,1),(-1,2),(-1,-2),(1,-2),(1,2),(2,-1),(2,1)]
def dfs(x,y,step,p,q,visited,ans):
    if step == p*q:
        return True
    for i in range(8):
        dx,dy = move[i]
        nx,ny =x+dx,y+dy
        if 0<=nx<q and 0<=ny<p and not visited[nx][ny]:
            visited[nx][ny] = True
            ans[step] = chr(nx+65) + str(ny+1)
            if dfs(nx,ny,step+1,p,q,visited,ans):
                return True
            visited[nx][ny] = False#回溯
    return False
n = int(input())
for m in range(1,n+1):
    p,q = map(int,input().split())
    ans = ["" for _ in range(p*q)]
    visited = [[False]*(p+1) for _ in range(q+1)]
    visited[0][0] = True
    ans[0] = "A1"
    if dfs(0,0,1,p,q,visited,ans):
        result = "".join(ans)
    else:
        result = "impossible"
    print(f"Scenario #{m}:")
    print(result)
    print()
```    
#骑士周游启发性关键算法
```python
def knight_graph(board_size):
    kt_graph = Graph()
    for row in range(board_size):           #遍历每一行
        for col in range(board_size):       #遍历行上的每一个格子
            node_id = pos_to_node_id(row, col, board_size) #把行、列号转为格子ID
            new_positions = gen_legal_moves(row, col, board_size) #按照 马走日，返回下一步可能位置
            for row2, col2 in new_positions:
                other_node_id = pos_to_node_id(row2, col2, board_size) #下一步的格子ID
                kt_graph.add_edge(node_id, other_node_id) #在骑士周游图中为两个格子加一条边
    return kt_graph


def knight_tour(n, path, u, limit):
    u.color = "gray"
    path.append(u)              #当前顶点涂色并加入路径
    if n < limit:
        neighbors = ordered_by_avail(u) #对所有的合法移动依次深入
        #neighbors = sorted(list(u.get_neighbors()))
        i = 0

        for nbr in neighbors:
            if nbr.color == "white" and \
                knight_tour(n + 1, path, nbr, limit):   #选择“白色”未经深入的点，层次加一，递归深入
                return True
        else:                       #所有的“下一步”都试了走不通
            path.pop()              #回溯，从路径中删除当前顶点
            u.color = "white"       #当前顶点改回白色
            return False
    else:
        return True

def ordered_by_avail(n):
    res_list = []
    for v in n.get_neighbors():
        if v.color == "white":
            c = 0
            for w in v.get_neighbors():
                if w.color == "white":
                    c += 1
            res_list.append((c,v))
    res_list.sort(key = lambda x: x[0])
    return [y[1] for y in res_list]

# class DFSGraph(Graph):
#     def __init__(self):
#         super().__init__()
#         self.time = 0                   #不是物理世界，而是算法执行步数
# 
#     def dfs(self):
#         for vertex in self:
#             vertex.color = "white"      #颜色初始化
#             vertex.previous = -1
#         for vertex in self:             #从每个顶点开始遍历
#             if vertex.color == "white":
#                 self.dfs_visit(vertex)  #第一次运行后还有未包括的顶点
#                                         # 则建立森林
# 
#     def dfs_visit(self, start_vertex):
#         start_vertex.color = "gray"
#         self.time = self.time + 1       #记录算法的步骤
#         start_vertex.discovery_time = self.time
#         for next_vertex in start_vertex.get_neighbors():
#             if next_vertex.color == "white":
#                 next_vertex.previous = start_vertex
#                 self.dfs_visit(next_vertex)     #深度优先递归访问
#         start_vertex.color = "black"
#         self.time = self.time + 1
#         start_vertex.closing_time = self.time
```
01094：整理一切
查看提交统计提问
总时间限制： 1000ms 内存限制： 65536kB
描述
不同值的升序排序序列是使用某种形式的小于运算符对元素从小到大进行排序的序列。例如，排序序列 A、B、C、D 意味着 A < B、B < C 和 C < D.在本题中，我们将为您提供一组形式为 A < B 的关系，并要求您确定是否指定了排序顺序。

输入
输入由多个问题实例组成。每个实例都以包含两个正整数 n 和 m 的行开头，第一个值表示要排序的对象数，其中 2 <= n <= 26。要排序的对象将是大写字母表的前 n 个字符。第二个值 m 表示在本问题实例中将给出的 A < B 形式的关系数。接下来是 m 行，每行包含一个由三个字符组成的此类关系：一个大写字母、字符“<”和第二个大写字母。任何字母都不会超出字母表的前 n 个字母的范围。n = m = 0 的值表示输入结束。
输出
对于每个问题实例，输出由一行组成。此行应为以下三行之一：

xxx 关系后确定的排序顺序：yyy...y.
无法确定排序顺序。
在 xxx 关系后发现不一致。

其中 xxx 是确定排序序列或发现不一致时处理的关系数，以先到者为准，yyy...y 是排序的升序序列。
样例输入
4 6
A<B
A<C
B<C
C<D
B<D
A<B
3 2
A<B
B<A
26 1
A<Z
0 0
样例输出
Sorted sequence determined after 4 relations: ABCD.
Inconsistency found after 2 relations.
Sorted sequence cannot be determined.
```python
from collections import deque
def topo_sort(graph):
    in_degree = {u:0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    q = deque([u for u in in_degree if in_degree[u] == 0])
    topo_order = [];flag = True
    while q:
        if len(q) > 1:
            flag = False#topo_sort不唯一确定
        u = q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    if len(topo_order) != len(graph): return 0
    return topo_order if flag else None
while True:
    n,m = map(int,input().split())
    if n == 0: break
    graph = {chr(x+65):[] for x in range(n)}
    edges = [tuple(input().split('<')) for _ in range(m)]
    for i in range(m):
        a,b = edges[i]
        graph[a].append(b)
        t = topo_sort(graph)
        if t:
            s = ''.join(t)
            print("Sorted sequence determined after {} relations: {}.".format(i+1,s))
            break
        elif t == 0:
            print("Inconsistency found after {} relations.".format(i+1))
            break
    else:
        print("Sorted sequence cannot be determined.")
```


#最大连通区域
```python
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


#bfs象棋，马
from collections import deque
sx,sy = map(int,input().split())
ex,ey = map(int,input().split())
blocks = set()
for _ in range(int(input())):
    blocks.add(tuple(map(int,input().split())))
dx = [-2,-2,-1,1,2,2,1,-1]
dy = [1,-1,-2,-2,1,-1,2,2]
q = deque()
q.append((sx,sy,f"({sx},{sy})"))
#查重
inQ = set()
inQ.add((sx,sy))
ans = 0
cur_path = ''
while q:
    tmp = deque()
    while q:
        x,y,path = q.popleft()
        wx,wy = [-1,0,1,0],[0,-1,0,1]
        if x == ex and y == ey:
            ans+=1
            if ans == 1:
                cur_path = path
        for i in range(8):
            nx,ny = x+dx[i],y+dy[i]
            hx,hy = x+wx[i//2],y+wy[i//2]
            if (nx,ny) != (sx,sy) and (hx,hy) not in blocks and 0<=nx<=10 and 0<=ny<=10:
                tmp.append((nx,ny,path+f"-({nx},{ny})"))

    if ans:
        break
    q=tmp
print(cur_path if ans == 1 else ans)
```
### 27635: 判断无向图是否连通有无回路(同23163)
http://cs101.openjudge.cn/practice/27635/
思路：
参考笔试题
代码
```python
# 
def build_graph():
    n,m = map(int,input().split())
    graph = {i:[] for i in range(n)}
    for _ in range(m):
        u,v = map(int,input().split())
        graph[u].append(v)
        graph[v].append(u)
    return graph,n
def check_connected(graph,n):
    visited = set()
    def dfs(u):
        nonlocal visited
        if u not in visited:
            visited.add(u)
            for v in graph[u]:
                dfs(v)
        return
    dfs(0)
    return len(visited) == n

def check_loop(graph,n):
    visited = [False for _ in range(n)]
    def dfs(u,f):
        visited[u] = True
        for v in graph[u]:
            if visited[v] == True:
                if v != f:
                    return True
            else:
                if dfs(v,u):
                    return True

    for i in range(n):
        if not visited[i]:
            if dfs(i,-1):
                return True
    return False


graph,n=build_graph()
if check_connected(graph,n):
    print('connected:yes')
else:
    print('connected:no')
if check_loop(graph,n):
    print("loop:yes ")
else:
    print("loop:no")
```


#有向图判断环或拓扑排序
```python
from collections import defaultdict
def dfs(node,color):
    color[node] = 1
    for neighbor in graph[node]:
        if color[neighbor] == 1:
            return True
        elif color[neighbor] == 0 and dfs(neighbor,color):
            return True
    color[node] = 2
    return False

T = int(input())
for _ in range(T):
    n,m = map(int,input().split())
    graph = defaultdict(list)
    for _ in range(m):
        x,y = map(int,input().split())
        graph[x].append(y)
    color=[0]*(n+1)
    is_cyclic = False
    for node in range(1,n+1):
        if color[node] == 0:
            if dfs(node,color):
                is_cyclic = True
                break
    print("Yes" if is_cyclic else "No")
```    
#

kruskal
```python
class UnionFind:
    def __init__(self,n):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self,x,y):
        px,py = self.find(x),self.find(y)
        if px != py:
            if self.rank[px] > self.rank[py]:
                self.parent[py] = px
            else:
                self.parent[px] = py
                if self.rank[px] == self.rank[py]:
                    self.rank[py] += 1
def kruskal(n,edges):
    uf = UnionFind(n)
    edges.sort(key = lambda x:x[2])
    mst,max_edge = 0,0
    for u,v,w in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u,v)
            mst += 1
            max_edge = max(max_edge,w)
            if mst == n-1:
                break

    return mst,max_edge
def main():
    n, m = map(int,input().split())
    edges = []
    for _ in range(m):
        u,v,c = map(int,input().split())
        edges.append((u-1,v-1,c))
#prim
import heapq
class edge:
    def __init__(self,ed,w):
        self.ed = ed
        self.w = w

n,m = map(int,input().split())
graph = {i:[] for i in range(1,n+1)}
for i in range(m):
    st,ed,w = map(int,input().split())
    graph[st].append(edge(ed,w))
    graph[ed].append(edge(st,w))
def prim(graph,n):
    cur_max = -1
    ans = -1

    heap = [(0,1)]

    visited = [False]*(n+1)
    #visited[1] = True
    while heap:
        w,node = heapq.heappop(heap)
        cur_max = max(cur_max, w)
        #可能后来bfs同层访问过
        if visited[node]:
            continue
        visited[node] = True
        ans += 1
        if ans == n-1:
            return ans, cur_max
        for u in graph[node]:
            if not visited[u.ed]:
                heapq.heappush(heap,(u.w,u.ed))

result = prim(graph,n)
print(*result)

    mst,max_edge = kruskal(n,edges)
    print(f"{mst} {max_edge}")
if __name__ == "__main__":
    main()
```    
#有钱数限制
```python
import heapq
class edge:
    def __init__(self,start,end,length,money):
        self.start = start
        self.end = end
        self.money = money
        self.length = length
k = int(input())
n = int(input())
r = int(input())
graph = {i:[] for i in range(1,n+1)}
for i in range(r):
    s,d,l,t = map(int,input().split())
    graph[s].append(edge(s,d,l,t))
def dijskra():
    visited=[0]*(n+1)
    ans=-1
    priorQueue=[]
    heapq.heappush(priorQueue,(0,0,1))#length,money,pos
    while priorQueue:
        length,money,pos = heapq.heappop(priorQueue)
        visited[pos] = 1
        if pos == n and money<=k:
            ans=length
            break
        if money > k:

            continue
        for road in graph[pos]:
            pos1 = road.end
            m1 = road.money+money
            l1 = road.length+length
            if m1<=k and visited[pos1] != 1:
                heapq.heappush(priorQueue,(l1,m1,pos1))
        visited[pos] = 0

    print(ans)
dijskra()
#含限制的最短路径算法+减枝
```

#可免单的DIJKSTRA
```python
import heapq
inf = float('inf')
n, m, k = map(int, input().split())  # 读取节点数量、边的数量和最大免单次数
graph = {i: [] for i in range(1, n + 1)}  # 初始化图的邻接表
vis = [[False] * (k + 1) for _ in range(n + 1)]  # 记录节点是否已经访问
dist = [[inf] * (k + 1) for _ in range(n + 1)]  # 记录从起点到每个节点的最短距离
# 读取边的信息并构建图的邻接表
for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))
    graph[v].append((u, w))

# Dijkstra算法求解带有免单功能的最短路径
#只记录最大值
def dijkstra(r=1):
    q = []
    dist[r][0] = 0
    heapq.heappush(q,(0,r,0))
    while q:
        cur_dist,pos,fre = heapq.heappop(q)
        if vis[pos][fre]:continue
        vis[pos][fre] = True
        for v,w in graph[pos]:
            #不免单，加入
            if dist[v][fre] > max(cur_dist,w):
                dist[v][fre] = max(cur_dist,w)
                heapq.heappush(q,(dist[v][fre],v,fre))
            #免单，加入
            if fre < k and dist[v][fre+1] > cur_dist:
                dist[v][fre+1] = dist[pos][fre]
                heapq.heappush(q,(dist[v][fre+1],v,fre+1))
dijkstra(1)
ans = inf
for i in range(k+1):
    ans = min(dist[n][i],ans)
print(ans if ans != inf else -1)
```
#python版
```python
#迷宫城堡Kosaraju
def dfs1(graph,node,visited,stack):
    visited[node] = True
    for nbr in graph[node]:
        if not visited[nbr]:
            dfs1(graph,nbr,visited,stack)
    stack.append(node)
def dfs2(graph,node,visited,component):
    visited[node] = True
    component.append(node)
    for nbr in graph[node]:
        if not visited[nbr]:
            dfs2(graph,nbr,visited,component)
def kosaraju(graph,n):
    stack = []
    visited = [False]*(n+1)
    for node in graph.keys():
        if not visited[node]:
            dfs1(graph, node, visited, stack)
    rG = {i:[] for i in range(1,n+1)}
    for node in range(1,len(graph)+1):
        for nbr in graph[node]:
            rG[nbr].append(node)
    visited = [False]*(n+1)
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(rG,node,visited,scc)
            sccs.append(scc)
    return sccs
def main(n,m):
    graph = {i:[] for i in range(1,n+1)}
    for _ in range(m):
        a,b = map(int,input().split())
        graph[a].append(b)
    sccs = kosaraju(graph,n)
    if len(sccs) == 1:
        print("Yes")
    else:
        print("No")

while True:
    n,m = map(int,input().split())
    if n==0 and m==0:
        break
    main(n,m)
#Tarjan算法
def Tarjan(graph):
    def dfs(node):
        nonlocal index,stack,indices,low_link,on_stack,sccs
        index+=1
        indices[node] = index
        low_link[node] = index
        stack.append(node)
        on_stack[node] = True
        for nbr in graph[node]:
            if indices[nbr] == 0:# Neighbor not visited yet
                dfs(nbr)
                low_link[node] = min(low_link[node],low_link[nbr])
            elif on_stack[nbr]:# Neighbor is in the current SCC
                low_link[node] = min(low_link[node], indices[nbr])
        if indices[node] == low_link[node]:
            scc = []
            while True:
                top = stack.pop()
                on_stack[top] = False
                scc.append(top)
                if top == node:
                    break
            sccs.append(scc)

    index = 0
    stack = []
    indices = [0]*(len(graph)+1)#次序
    low_link = [0]*(len(graph)+1)
    on_stack = [False]*(len(graph)+1)
    sccs = []
    for node in range(1,len(graph)+1):
        if indices[node] == 0:
            dfs(node)
    return sccs
def main(n,m):
    graph = {i:[] for i in range(1,n+1)}
    for _ in range(m):
        a,b = map(int,input().split())
        graph[a].append(b)
    sccs = Tarjan(graph)
    if len(sccs) == 1:
        print("Yes")
    else:
        print("No")

while True:
    n,m = map(int,input().split())
    if n==0 and m==0:
        break
    main(n,m)

#地铁：dijskra
import math
import heapq

def get_distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)
sx,sy,ex,ey = map(int,input().split())
min_time = {}
rails = set()
while True:
    try:
        rail = list(map(int,input().split()))
        if not rail:
            break
        if rail == [-1,-1]:
            break
        stations = [(rail[2*i],rail[2*i+1]) for i in range(len(rail)//2-1)]
        for j,station in enumerate(stations):
            min_time[station] = float("inf")
            if j != len(stations)-1:
                rails.add((station,stations[j+1]))
                rails.add((stations[j+1],station))
    except EOFError:
        break
min_time[(sx,sy)],min_time[(ex,ey)] =0,float("inf")
min_heap = [(0,sx,sy)]
while min_heap:
    cur_time,x,y = heapq.heappop(min_heap)
    if cur_time > min_time[(x,y)]:
        continue
    if (x,y) == (ex,ey):
        break
    for position in min_time.keys():
        if position == (x,y):
            continue
        nx,ny = position
        dis = get_distance(x,y,nx,ny)
        rail_factor = 4 if (position,(x,y)) in rails or ((x,y),position) in rails else 1
        new_time = cur_time + dis/(10000*rail_factor)
        if new_time < min_time[position]:
            min_time[position] = new_time
```            

P1260:火星大工程
查看提交统计提问
总时间限制: 1000ms 内存限制: 65536kB
描述
中国要在火星上搞个大工程，即建造n个科考站

建科考站需要很专业的设备，不同的科考站需要不同的设备来完成

有的科考站必须等另外一些科考站建好后才能建。

每个设备参与建完一个科考站后，都需要一定时间来保养维修，才能参与到下一个科考站的建设。

所以，会发生科考站A建好后，必须至少等一定时间才能建科考站B的情况。因为B必须在A之后建，且建B必需的某个设备，参与了建A的工作，它需要一定时间进行维修保养。

一个维修保养任务用三个数a b c表示，意即科考站b必须等a建完才能建。而且，科考站a建好后，建a的某个设备必须经过时长c的维修保养后，才可以开始参与建科考站b。

假设备都很牛，只要设备齐全可用，建站飞快就能完成，建站时间忽略不计。一开始所有设备都齐全可用。

给定一些维修保养任务的描述，求所有科考站都建成，最快需要多长时间。

有的维修保养任务，能开始的时候也可以先不开始，往后推迟一点再开始也不会影响到整个工期。问在不影响最快工期的情况下，哪些维修保养任务的开始时间必须是确定的。按字典序输出这些维修保养工任务，输出的时候不必输出任务所需的时间。
输入
第一行两个整数n,m，表示有n个科考站，m个维修保养任务。科考站编号为1，2.....n
接下来m行，每行三个整数a b c，表示一个维修保养任务
1 < n,m <=3000
输出
先输出所有科考站都建成所需的最短时间
然后按字典序输出开始时间必须确定的维修保养任务
样例输入
9 11
1 2 6
1 3 4
1 4 5
2 5 1
3 5 1
4 6 2
5 7 9
5 8 7
6 8 4
7 9 2
8 9 4
样例输出
18
1 2
2 5
5 7
5 8
7 9
8 9
#关键路径
```python
from collections import defaultdict,deque
class Edge:
    def __init__(self,end,weight):
        self.end = end
        self.weight = weight
    def __lt__(self,other):
        return self.end < other.end
def find_critical_activities(n,m,edges):
    graph = defaultdict(list)
    in_degree = [0]*n
    for s,e,w in edges:
        graph[s-1].append(Edge(e-1,w))
        in_degree[e-1] += 1
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topological_order = []
    while queue:
        node = queue.popleft()
        topological_order.append(node)
        for edge in graph[node]:
            in_degree[edge.end] -= 1
            if in_degree[edge.end] == 0:
                queue.append(edge.end)
    # 计算最早开始时间
    earliest = [0]*n
    for i in topological_order:
        for edge in graph[i]:
            earliest[edge.end] = max(earliest[edge.end],earliest[i]+edge.weight)
    T = max(earliest)
    latest = [T]*n
    # 计算最晚开始时间
    for j in reversed(topological_order):
        for edge in graph[j]:
            latest[j] = min(latest[j],latest[edge.end]-edge.weight)
    critical_events = [i for i in range(n) if earliest[i] == latest[i]]
    critical_activities = []
    for i in critical_events:
        graph[i].sort()
        for edge in graph[i]:
            if edge.end in critical_events and earliest[edge.end] - earliest[i] == edge.weight:
                critical_activities.append((i+1,edge.end+1))
    return T,critical_activities


n,m = map(int,input().split())
edges = [list(map(int,input().split())) for _ in range(m)]
T,critical_activities = find_critical_activities(n,m,edges)
print(T)
#print(critical_activities)
for a in critical_activities:
    print(*a)


            heapq.heappush(min_heap,(new_time,nx,ny))
print(round(min_time[(ex,ey)]*60))
```


## 3. 课程总结

如果愿意，请同学或多或少做一个本门课程的学习总结。便于之后师弟师妹跟进学习，也便于教师和助教改进教学。例如：分享自己的学习心得、笔记。



## 参考

1.科学上网 Scientific Internet

北大学长提供的Clash，请自己取用。
https://189854.xyz/verify/
https://blog.189854.xyz/blog/walless/2023/11/04/clash.html



2.图床，把图片放到云上去，而不是本地的意思。如果设置图床，分享md文件，其他人也能看到图片；否则因为md嵌入的图片在本地，只有编辑者能看到；后者的情况解决方法还可以是导出包含图片的pdf文件分享。图床如果是免费的，过一阵可能会失效，之前用过非github的免费图床，导致链接失效了。github是免费的，目前比较稳定。

1）Typora + GitHub = 效率，https://mp.weixin.qq.com/s/hmkGZln-xatrWrBZrY9t-g

2）Typora+PicGo+Github解决个人博客图片上传问题 https://zhuanlan.zhihu.com/p/367529569

3）设置的图床目录是Public

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240228102834113.png" alt="image-20240228102834113" style="zoom:33%;" />



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240228102902018.png" alt="image-20240228102902018" style="zoom:33%;" />





3.Github图片不显示，原因是DNS污染。两种解决方法，或者直接添加1）给出的ip列表，或者2）自己找出ip添加。

1）Github图片显示不出来？两步解决！ https://zhuanlan.zhihu.com/p/345258967?utm_id=0&wd=&eqid=ce16938700061ac4000000056470d782 。

2）https://www.ipaddress.com查到ip，添加到hosts后，在移动宽带网络中，可以显示md中的图片。 参考：解决raw.githubusercontent.com无法访问的问题（picgo+github配置图床图片不显示，但仓库已存储成功），https://blog.51cto.com/reliableyang/6457392.  



