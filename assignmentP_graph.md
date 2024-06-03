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
###22460:火星车勘探
查看提交统计提问
总时间限制: 1000ms 内存限制: 65535kB
描述
火星这颗自古以来寄托了中国人无限遐思的红色星球，如今第一次留下了中国人的印迹。2021年5月15日，“天问一号”探测器成功降落在火星预选着陆区。这标志着中国首次火星探测着陆任务取得成功，同时也使中国成为继美国之后第二个实现探测器着陆火星的国家。

假设火星车需要对形如二叉树的地形进行遍历勘察。火星车初始处于二叉树地形的根节点，对二叉树进行前序遍历。当火星车遇到非空节点时，则采样一定量的泥土样本，记录下样本数量；当火星车遇到空节点，使用一个标记值#进行记录。



对上面的二叉树地形可以前序遍历得到 9 3 4 # # 1 # # 2 # 6 # #，其中 # 代表一个空节点，整数表示在该节点采样的泥土样本数量。

我们的任务是，给定一串以空格分隔的序列，验证它是否是火星车对于二叉树地形正确的前序遍历结果。

输入
每组输入包含多个测试数据，每个测试数据由两行构成。
每个测试数据的第一行：1个正整数N，表示遍历结果中的元素个数。
每个测试数据的第二行：N个以空格分开的元素，每个元素可以是#，也可以是小于100的正整数。(1<=N<=200000)
输入的最后一行为0，表示输入结束。
输出
对于每个测试数据，输出一行判断结果。
输入的序列如果是对某个二叉树的正确的前序遍历结果，则输出“T”，否则输出“F”。
样例输入
13
9 3 4 # # 1 # # 2 # 6 # #
4
9 # # 1
2
# 99
0
样例输出
T
F
F
```python
class Node:
    def __init__(self,key):
        self.key = key
        self.left = None
        self.right = None
#根据前序建树
def build_tree(s):
    root = Node(s.pop())
    if root.key != "#":
        root.left = build_tree(s)
        root.right = build_tree(s)
    return root
while True:
    try:
        n = int(input())
        if n == 0:
            break
        s = input().split()[::-1]
        _ = build_tree(s)
        print("F" if len(s) else "T")
    except IndexError:
        print("F")
```
###02775:文件结构“图”
查看提交统计提问
总时间限制: 1000ms 内存限制: 65536kB
描述
在计算机上看到文件系统的结构通常很有用。Microsoft Windows上面的"explorer"程序就是这样的一个例子。但是在有图形界面之前，没有图形化的表示方法的，那时候最好的方式是把目录和文件的结构显示成一个"图"的样子，而且使用缩排的形式来表示目录的结构。比如：


ROOT
|     dir1
|     file1
|     file2
|     file3
|     dir2
|     dir3
|     file1
file1
file2
这个图说明：ROOT目录包括三个子目录和两个文件。第一个子目录包含3个文件，第二个子目录是空的，第三个子目录包含一个文件。

输入
你的任务是写一个程序读取一些测试数据。每组测试数据表示一个计算机的文件结构。每组测试数据以'*'结尾，而所有合理的输入数据以'#'结尾。一组测试数据包括一些文件和目录的名字（虽然在输入中我们没有给出，但是我们总假设ROOT目录是最外层的目录）。在输入中,以']'表示一个目录的内容的结束。目录名字的第一个字母是'd'，文件名字的第一个字母是'f'。文件名可能有扩展名也可能没有（比如fmyfile.dat和fmyfile）。文件和目录的名字中都不包括空格,长度都不超过30。一个目录下的子目录个数和文件个数之和不超过30。
输出
在显示一个目录中内容的时候，先显示其中的子目录（如果有的话），然后再显示文件（如果有的话）。文件要求按照名字的字母表的顺序显示（目录不用按照名字的字母表顺序显示，只需要按照目录出现的先后显示）。对每一组测试数据，我们要先输出"DATA SET x:"，这里x是测试数据的编号（从1开始）。在两组测试数据之间要输出一个空行来隔开。

你需要注意的是，我们使用一个'|'和5个空格来表示出缩排的层次。
样例输入
file1
file2
dir3
dir2
file1
file2
]
]
file4
dir1
]
file3
*
file2
file1
*
#
样例输出
DATA SET 1:
ROOT
|     dir3
|     |     dir2
|     |     file1
|     |     file2
|     dir1
file1
file2
file3
file4

DATA SET 2:
ROOT
file1
file2
提示
一个目录和它的子目录处于不同的层次
一个目录和它的里面的文件处于同一层次
```python
from sys import exit
def print_dirs(root):
    tap = "|     "
    print(tap*root.level+root.name)
    for nbr in root.dirs:
        print_dirs(nbr)
    for file in sorted(root.files):
        print(tap*root.level+file)
    return
class dirs:
    def __init__(self,name,level):
        self.name = name
        self.dirs = []
        self.files = []
        self.level = level
flag = True
count = 0
ans = []
while flag:
    count+=1
    root = dirs("ROOT",0)
    stack_dirs = [root]
    while (s := input()) != "*":
        if s[0] == "f":
            if stack_dirs:
                stack_dirs[-1].files.append(s)
        if s[0] == "d":
          new_dir = dirs(s,stack_dirs[-1].level+1)
          stack_dirs[-1].dirs.append(new_dir)
          stack_dirs.append(new_dir)
        if s == "]" and stack_dirs:
            stack_dirs.pop()
        if s == "#":
            flag = False
            exit(0)
    print(f"DATA SET {count}:")
    print_dirs(root)
    print()
```
04082:树的镜面映射
查看提交统计提问
总时间限制: 1000ms 内存限制: 65536kB
描述
一棵树的镜面映射指的是对于树中的每个结点，都将其子结点反序。例如，对左边的树，镜面映射后变成右边这棵树。

    a                             a
  / | \                         / | \
 b  c  f       ===>            f  c  b
   / \                           / \
  d   e                         e   d
我们在输入输出一棵树的时候，常常会把树转换成对应的二叉树，而且对该二叉树中只有单个子结点的分支结点补充一个虚子结点“$”，形成“伪满二叉树”。

例如，对下图左边的树，得到下图右边的伪满二叉树

    a                             a
  / | \                          / \
 b  c  f       ===>             b   $
   / \                         / \
  d   e                       $   c                          
                                 / \
                                d   f
                               / \
                              $   e
然后对这棵二叉树进行前序遍历，如果是内部结点则标记为0，如果是叶结点则标记为1，而且虚结点也输出。

现在我们将一棵树以“伪满二叉树”的形式输入，要求输出这棵树的镜面映射的宽度优先遍历序列。

输入
输入包含一棵树所形成的“伪满二叉树”的前序遍历。
第一行包含一个整数，表示结点的数目。
第二行包含所有结点。每个结点用两个字符表示，第一个字符表示结点的编号，第二个字符表示该结点为内部结点还是外部结点，内部结点为0，外部结点为1。结点之间用一个空格隔开。
数据保证所有结点的编号都为一个小写字母。
输出
输出包含这棵树的镜面映射的宽度优先遍历序列，只需要输出每个结点的编号，编号之间用一个空格隔开。
样例输入
9
a0 b0 $1 c0 d0 $1 e1 f1 $1
样例输出
a f c b e d
#树的镜像转换
```python
class TreeNode:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
        self.children = []
        self.parent = None
#用栈前序建树
stack = []
nodes = []
n = int(input())
alist = list(input().split())
for x in alist:
    temp = TreeNode(x[0])
    nodes.append(temp)
    if stack:
        if stack[-1].left:
            stack[-1].right = temp
            stack.pop()
        else:
            stack[-1].left = temp
    if x[1] == "0":
        stack.append(temp)
#2-->多：左孩子右兄弟
for x in nodes:
    if x.left and x.left.value != "$":
        x.children.append(x.left)
        x.left.parent = x
    if x.right and x.right.value != "$":
        x.parent.children.append(x.right)
        x.right.parent = x.parent
for x in nodes:
    x.children = x.children[::-1]
lst1 = [nodes[0]]
for x in lst1:
    if x.children:
        lst1+=x.children
print(" ".join([x.value for x in lst1]))
```

#括号嵌套树
```python
class TreeNode:
    def __init__(self,value):
        self.value = value
        self.children = []
def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():
            node = TreeNode(char)
            if stack:
                stack[-1].children.append(node)
        elif char == '(':
            if node:
                stack.append(node)
        elif char == ')':
            if stack:
                node = stack.pop()
    return node
def preorder(node):
    output = [node.value]
    for c in node.children:
        output.extend(preorder(c))
    return ''.join(output)
def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)
def main():
    s = input().strip()
    s = ''.join(s.split())
    root = parse_tree(s)
    if root:
        print(preorder(root))
        print(postorder(root))
    else:
        print("input tree string error!")
if __name__ == "__main__":
    main()
```
###03720:文本二叉树
查看提交统计提问
总时间限制: 1000ms 内存限制: 65536kB
描述
如上图，一棵每个节点都是一个字母，且字母互不相同的二叉树，可以用以下若干行文本表示:
A
-B
--*
--C
-D
--E
---*
---F
在这若干行文本中：

1) 每个字母代表一个节点。该字母在文本中是第几行，就称该节点的行号是几。根在第1行
2) 每个字母左边的'-'字符的个数代表该结点在树中的层次（树根位于第0层）
3) 若某第 i 层的非根节点在文本中位于第n行，则其父节点必然是第 i-1 层的节点中，行号小于n,且行号与n的差最小的那个
4) 若某文本中位于第n行的节点(层次是i) 有两个子节点，则第n+1行就是其左子节点，右子节点是n+1行以下第一个层次为i+1的节点
5) 若某第 i 层的节点在文本中位于第n行，且其没有左子节点而有右子节点，那么它的下一行就是 i+1个'-' 字符再加上一个 '*'


给出一棵树的文本表示法，要求输出该数的前序、后序、中序遍历结果
输入
第一行是树的数目 n

接下来是n棵树，每棵树以'0'结尾。'0'不是树的一部分
每棵树不超过100个节点
输出
对每棵树，分三行先后输出其前序、后序、中序遍历结果
两棵树之间以空行分隔
样例输入
2
A
-B
--*
--C
-D
--E
---*
---F
0
A
-B
-C
0
样例输出
ABCDEF
CBFEDA
BCAEFD

ABC
BCA
BAC
```python
#数组存树+遍历
class Node:
    def __init__(self,value,depth):
        self.value = value
        self.depth = depth
        self.left = None
        self.right = None
    def preorder_traversal(self):
        nodes = [self.value]
        if self.left and self.left.value != '*':
            nodes += self.left.preorder_traversal()
        if self.right and self.right.value != '*':
            nodes += self.right.preorder_traversal()
        return nodes
    def inorder_traversal(self):
        nodes = []
        if self.left and self.left.value != '*':
            nodes += self.left.inorder_traversal()
        nodes.append(self.value)
        if self.right and self.right.value != '*':
            nodes += self.right.inorder_traversal()
        return nodes
    def postorder_traversal(self):
        nodes = []
        if self.left and self.left.value != '*':
            nodes += self.left.postorder_traversal()

        if self.right and self.right.value != '*':
            nodes += self.right.postorder_traversal()
        nodes.append(self.value)
        return nodes


def build_tree():
    n = int(input())
    for _ in range(n):
        tree = []
        stack = []
        while True:
            s = input()
            if s == '0':
                break
            depth = len(s)-1
            node = Node(s[-1],depth)
            tree.append(node)
            # Finding the parent for the current node
            #stack为单调栈存索引
            while stack and tree[stack[-1]].depth >= depth:
                stack.pop()
            if stack:
                parent = tree[stack[-1]]
                if not parent.left:
                    parent.left = node
                else:
                    parent.right = node
            stack.append(len(tree)-1)
        yield tree[0]
for root in build_tree():
    print("".join(root.preorder_traversal()))
    print("".join(root.postorder_traversal()))
    print("".join(root.inorder_traversal()))
    print()
```
```python
05430:表达式·表达式树·表达式求值
查看提交统计提问
总时间限制: 1000ms 内存限制: 65535kB
描述
众所周知，任何一个表达式，都可以用一棵表达式树来表示。例如，表达式a+b*c，可以表示为如下的表达式树：

   +
  / \
a   *
    / \
    b c

现在，给你一个中缀表达式，这个中缀表达式用变量来表示（不含数字），请你将这个中缀表达式用表达式二叉树的形式输出出来。

输入
输入分为三个部分。
第一部分为一行，即中缀表达式(长度不大于50)。中缀表达式可能含有小写字母代表变量（a-z），也可能含有运算符（+、-、*、/、小括号），不含有数字，也不含有空格。
第二部分为一个整数n(n < 10)，表示中缀表达式的变量数。
第三部分有n行，每行格式为C　x，C为变量的字符，x为该变量的值。
输出
输出分为三个部分，第一个部分为该表达式的逆波兰式，即该表达式树的后根遍历结果。占一行。
第二部分为表达式树的显示，如样例输出所示。如果该二叉树是一棵满二叉树，则最底部的叶子结点，分别占据横坐标的第1、3、5、7……个位置（最左边的坐标是1），然后它们的父结点的横坐标，在两个子结点的中间。如果不是满二叉树，则没有结点的地方，用空格填充（但请略去所有的行末空格）。每一行父结点与子结点中隔开一行，用斜杠（/）与反斜杠（\）来表示树的关系。/出现的横坐标位置为父结点的横坐标偏左一格，\出现的横坐标位置为父结点的横坐标偏右一格。也就是说，如果树高为m，则输出就有2m-1行。
第三部分为一个整数，表示将值代入变量之后，该中缀表达式的值。需要注意的一点是，除法代表整除运算，即舍弃小数点后的部分。同时，测试数据保证不会出现除以0的现象。
import operator as op
class Node:
    def __init__(self,x):
        self.value = x
        self.left = None
        self.right = None
dict_priority={"*":2,"/":2,"+":1,"-":1,")":0,"(":0}
def infix_trans(infix):
    postfix = []
    op_stack = []
    for char in infix:
        if char.isalpha():
            postfix.append(char)
        else:
            if char == '(':
                op_stack.append(char)
            elif char == ')':
                while op_stack and op_stack[-1] != '(':
                    postfix.append(op_stack.pop())
                op_stack.pop()
            else:
                while op_stack and dict_priority[op_stack[-1]]>=dict_priority[char] and op_stack != '(':
                    postfix.append(op_stack.pop())
                op_stack.append(char)
    while op_stack:
        postfix.append(op_stack.pop())
    return postfix
def build_tree(postfix):
    stack = []
    for item in postfix:
        if item in '+-*/':
            node = Node(item)
            node.right = stack.pop()
            node.left = stack.pop()
        else:
            node = Node(item)
        stack.append(node)
    return stack[0]
def get_val(expr_tree,var_vals):
    if expr_tree.value in '+-*/':
        operator = {'+':op.add,'-':op.sub,'*':op.mul,'/':op.floordiv}
        return operator[expr_tree.value](get_val(expr_tree.left,var_vals),get_val(expr_tree.right,var_vals))
    else:
        return var_vals[expr_tree.value]
def getDepth(tree_root):
    left_depth = getDepth(tree_root.left) if tree_root.left else 0
    right_depth = getDepth(tree_root.right) if tree_root.right else 0
    return max(left_depth,right_depth) + 1
def printExpressionTree(tree_root,d):
    graph = [" "*(2**d-1)+tree_root.value + " "*(2**d-1)]
    graph.append(" "*(2**d-2)+("/" if tree_root.left else " ")+
                 " "+("\\"if tree_root.right else " ")+" "*(2**d-2))
    if d == 0:
        return tree_root.value
    d-=1
    if tree_root.left:
        left = printExpressionTree(tree_root.left,d)
    else:
        left = [" "*(2**(d+1)-1)]*(2*d+1)
    right = printExpressionTree(tree_root.right,d) if tree_root.right else [
        " "*(2**(d+1)-1)]*(2*d+1)
    for i in range(2*d+1):
        graph.append(left[i] + " " + right[i])
    return graph

infix = input().strip()
n = int(input())
vars_value={}
for i in range(n):
    char,num = input().split()
    vars_value[char] = int(num)
postfix = infix_trans(infix)
tree_root = build_tree(postfix)
print(''.join(str(x) for x in postfix))
expression_value = get_val(tree_root, vars_value)


for line in printExpressionTree(tree_root, getDepth(tree_root)-1):
    print(line.rstrip())


print(expression_value)
```python
27638:求二叉树的高度和叶子数目
查看提交统计提问
总时间限制: 1000ms 内存限制: 65536kB
描述
给定一棵二叉树，求该二叉树的高度和叶子数目二叉树高度定义：从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的结点数减1为树的高度。只有一个结点的二叉树，高度是0。

输入
第一行是一个整数n，表示二叉树的结点个数。二叉树结点编号从0到n-1，根结点n <= 100 接下来有n行，依次对应二叉树的编号为0,1,2....n-1的节点。 每行有两个整数，分别表示该节点的左儿子和右儿子的编号。如果第一个（第二个）数为-1则表示没有左（右）儿子
输出
在一行中输出2个整数，分别表示二叉树的高度和叶子结点个数
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
n = int(input())
exist_parents = [False]*n
forest = [TreeNode() for i in range(n)]
root_index = 0
def tree_height(root):
    if not root:
        return -1
    else:
        return 1 + max(tree_height(root.left),tree_height(root.right))
for i in range(n):
    left,right = map(int,input().split())
    if right != -1:
        forest[i].right = forest[right]
        exist_parents[right] = True
    if left != -1:
        forest[i].left = forest[left]
        exist_parents[left] = True
for i in range(n):
    if not exist_parents[i]:
        root_index = i
        break
def count_leaves(root):
    if not root:
        return 0
    if root.left is None and root.right is None:
        return 1
    else:
        return count_leaves(root.left) + count_leaves(root.right)
height = tree_height(forest[root_index])
leaves = count_leaves(forest[root_index])
print(f'{height} {leaves}')
```

```python
#前缀树，26叉树
class TrieNode:
    def __init__(self):
        self.child = {}
        self.is_end = 0
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self,nums):
        curNode = self.root
        for x in nums:
            if x not in curNode.child:
                curNode.child[x] = TrieNode()
            curNode = curNode.child[x]
        curNode.is_end = 1

    def search(self,num):
        curnode = self.root
        for x in num:
            if x not in curnode.child:
                return 0
            curnode = curnode.child[x]
        return 1
t = int(input())
p=[]
for _ in range(t):
    n = int(input())
    nums = []
    for _ in range(n):
        nums.append(str(input()))
    nums.sort(reverse=True)
    s = 0
    trie = Trie()
    for num in nums:
        s += trie.search(num)
        trie.insert(num)
    if s > 0:
        print('NO')
    else:
        print('YES')
```
```
#树序转换  前中-->后
class TreeNode:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
def build_tree(preorder,inorder):
    if not preorder or not inorder:
        return None
    root = TreeNode(preorder[0])
    root_index = inorder.index(root.value)
    root.left = build_tree(preorder[1:1+root_index],inorder[:root_index])
    root.right = build_tree(preorder[1+root_index:],inorder[root_index+1:])
    return root
def postTraverse(root):
    res=[]
    if root.left:
        res.extend(postTraverse(root.left))
    if root.right:
        res.extend(postTraverse(root.right))
    res.append(root.value)
    return res

def main():
    preorder = list(input())
    inorder = list(input())
    root = build_tree(preorder,inorder)
    print(''.join(postTraverse(root)))


while True:
    try:
        main()
    except EOFError:
        break
```
01760：磁盘树
查看提交统计提问
总时间限制： 1000ms 内存限制： 65536kB
描述
黑客比尔不小心丢失了工作站硬盘上的所有信息，并且他没有其内容的备份副本。他并不后悔丢失文件本身，而是后悔他在多年的工作中创建和珍惜的非常漂亮和方便的目录结构。幸运的是，比尔的硬盘上有几份目录列表的副本。使用这些列表，他能够恢复某些目录的完整路径（如“WINNT\SYSTEM32\CERTSRV\CERTCO~1\X86”）。他把他找到的每条路径都写在一个单独的行上，把它们都放在一个文件中。你的任务是编写一个程序，通过提供格式良好的目录树来帮助 Bill 恢复他最先进的目录结构。
输入
输入文件的第一行包含单个整数 N （1 <= N <= 500），表示不同目录路径的总数。然后是带有目录路径的 N 行。每个目录路径占用一行，不包含任何空格，包括前导或尾随空格。任何路径都不超过 80 个字符。每个路径列出一次，由多个目录名称组成，这些目录名称由反斜杠 （“\”） 分隔。

每个目录名称由 1 到 8 个大写字母、数字或以下列表中的特殊字符组成：感叹号、数字符号、美元符号、百分号、与号、撇号、左括号和右括号、连字符号、商业 at、回旋重音、下划线、重音、左大括号和右大括号以及波浪号 （“！#$%&'（）-@^_'{}~”）。
输出
将格式化的目录树写入输出文件。每个目录名称都应列在自己的行上，前面有若干空格，以指示其在目录层次结构中的深度。子目录应按词典顺序列在其父目录之后，前面应比其父目录多一个空格。顶级目录的名称前不得标有空格，并应按词典顺序列出。有关输出格式的说明，请参阅下面的示例。
样例输入
7
WINNT\SYSTEM32\CONFIG
GAMES
WINNT\DRIVERS
HOME
WIN\SOFT
GAMES\DRIVERS
WINNT\SYSTEM32\CERTSRV\CERTCO~1\X86
样例输出
GAMES
 DRIVERS
HOME
WIN
 SOFT
WINNT
 DRIVERS
 SYSTEM32
  CERTSRV
   CERTCO~1
    X86
  CONFIG
```python
#磁盘树
class Node:
    def __init__(self):
        self.children={}
class Trie:
    def __init__(self):
        self.root = Node()
    def insert(self,w):
        cur=self.root
        for u in w.split('\\'):
            if u not in cur.children:
                cur.children[u]=Node()
            cur = cur.children[u]
    def dfs(self,a,layer):
        for c in sorted(a.children):
            print(' '*layer+c)
            self.dfs(a.children[c],layer+1)
s=Trie()
for _ in range(int(input())):
    x=input()
    s.insert(x)
s.dfs(s.root,0)

#并查集
def find(x):#压缩路径
    if x != s[x]:
        s[x] = find(s[x])
    return s[x]

def union(x,y):
    rootx = find(x);rooty = find(y)
    if rootx != rooty:
        s[rooty] = rootx
    return
def is_connected(x,y):

    return find(x) == find(y)
T = int(input())
for i in range(1,T+1):

    index = i
    flag = True
    n,m = map(int,input().split())
    if n==1:
        print("Scenario #{}:".format(index))
        print("No suspicious bugs found!")
        print()
        continue
    s=[i for i in range(2*n+2)]
    for i in range(m):
        t1,t2 = map(int,input().split())
        if not flag:
            continue
        if is_connected(t1,t2):
            flag = False

        union(t1,t2+n)
        union(t1+n,t2)
    print("Scenario #{}:".format(index))
    print("Suspicious bugs found!" if not flag else "No suspicious bugs found!")
    print()
```
```python
#败方树
from collections import deque
class TreeNode:
    def __init__(self,value,min_win):
        self.value = value
        self.min_win = min_win
        self.left = None
        self.right = None
def build_tree(values):
    queue = deque(TreeNode(value,value) for value in values)
    while len(queue) > 1:
        left_node = queue.popleft()
        right_node = queue.popleft()
        new_node = TreeNode(max(left_node.min_win,right_node.min_win),min(left_node.min_win,right_node.min_win))
        new_node.left = left_node
        new_node.right = right_node
        queue.append(new_node)
    root = TreeNode(queue[0].min_win,queue[0].min_win)
    root.left = queue[0]
    return root
def show(n,root):
    queue = deque([root])
    result = []
    while queue:
        if len(result) == n:
            print(*result)
            return
        cur = queue.popleft()
        result.append(cur.value)
        if cur.left:
            queue.append(cur.left)
        if cur.right:
            queue.append(cur.right)
n,m = map(int,input().split())
initial_values = list(map(int,input().split()))
root = build_tree(initial_values)
show(n,root)
for _ in range(m):
    pos,value = map(int,input().split())
    initial_values[pos] = value
    root = build_tree(initial_values)
    show(n,root)
```

27947: 动态中位数
http://cs101.openjudge.cn/practice/27947/

思路： 用堆

代码
```python
# 
import heapq
def dynamic_median(nums):
    min_heap = []#存较大数据
    max_heap = []#存较小数据
    median = []
    for i,num in enumerate(nums):
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap,-num)
        else:
            heapq.heappush(min_heap,num)
        if len(max_heap)-len(min_heap)>1:
            heapq.heappush(min_heap,-heapq.heappop(max_heap))
        elif len(min_heap)>len(max_heap):
            heapq.heappush(max_heap,-heapq.heappop(min_heap))
        if i%2 == 0:
            median.append(-max_heap[0])
    return median

def main():
    nums = list(map(int,input().split()))
    median = dynamic_median(nums)
    print(len(median))
    print(*median)
T = int(input())
for _ in range(T):
    main()
```
体育老师组织学生进行跳高训练，查看其相对于上一次训练中跳高的成绩是否有所进步。为此，他组织同学们按学号排成一列进行测试。本次测验使用的老式测试仪，只能判断同学跳高成绩是否高于某一预设值，且由于测试仪器构造的问题，其横杠只能向上移动。由于老师只关心同学是否取得进步，因此老师只将跳高的横杠放在该同学上次跳高成绩的位置，查看同学是否顺利跃过即可。为了方便进行上次成绩的读取，同学们需按照顺序进行测验，因此对于某个同学，当现有的跳高测试仪高度均高于上次该同学成绩时，体育老师需搬出一个新的测试仪进行测验。已知同学们上次测验的成绩，请问体育老师至少需要使用多少台测试仪进行测验？
由于采用的仪器精确度很高，因此测试数据以毫米为单位，同学们的成绩为正整数，最终测试数据可能很大，但不超过10000，且可能存在某同学上次成绩为0。
输入
输入共两行，第一行为一个数字N，N<=100000，表示同学的数量。第二行为N个数字，表示同学上次测验的成绩（从1号到N号排列）。
输出
一个正整数，表示体育老师最少需要的测试仪数量。
样例输入
5
1 7 3 5 2
样例输出
3
#二分+贪心
```python
from bisect import *
cur_temps = []
N = int(input())
scores = list(map(int,input().split()))
for idx in range(N):
    cur = scores[idx]
    if cur_temps:
        if cur >= cur_temps[-1]:
            cur_temps[-1] = cur
        else:
            ind = bisect(cur_temps,cur)#插入位置,同取右
            if ind == 0:
                cur_temps.insert(0,cur)
            else:
                cur_temps[ind-1] = cur
    else:
        cur_temps.append(cur)
print(len(cur_temps))
```
//AC自动机
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 1e6+5;
struct node{
    int son[26];
    int end;
    int fail;
}t[N];//字典树（数组存）
int cnt;//字典树在数组中的位置
void Insert(char *s){
    int now = 0;
    for (int i=0;s[i];i++){
        int ch = s[i] - 'a';
        if(t[now].son[ch]==0)
            t[now].son[ch] = cnt++;
        now = t[now].son[ch];    
    }
    t[now].end++;//?
}
//bfs建立指针
void getFail(){
    queue<int>q;
    for(int i=0;i<26;i++){
        if(t[0].son[i]) q.push(t[0].son[i]);
    }
    //fail-->同义词
    while (!q.empty()){
        int now = q.front();
        q.pop();
        for(int i=0;i<26;i++){
            if (t[now].son[i]){
                t[t[now].son[i]].fail = t[t[now].fail].son[i];
                q.push(t[now].son[i]);
               }
            else
                t[now].son[i] = t[t[now].fail].son[i];//虚拟节点   
        }
    }
}
int query(char *s){
    int ans = 0;
    int now = 0;
    for(int i=0;s[i];i++){
        int ch = s[i]-'a';
        now = t[now].son[ch];
        int tmp = now;
        while (tmp&&t[tmp].end!=-1){
            ans+=t[tmp].end;
            t[tmp].end = -1;
            tmp = t[tmp].fail;
            //cout<<"tmp="<<tmp<<" "<<t[tmp].son;
    }
    }
    return ans;
}
char str[N];
int main(){
    int k; scanf("%d",&k);
    while (k--){
        memset(t,0,sizeof(t));
        cnt = 1;
        int n; scanf("%d",&n);
        while (n--){
            scanf("%s",str);
            Insert(str);
            }
        getFail();
        scanf("%s",str);
        printf("%d\n",query(str));    
    }
    return 0;

}
#八皇后
```python
#递归回溯
def solve_n_queen(n):
    solutions = []
    queens = [-1] * n    #queens = [-1] * n  # 存储每一行皇后所在的列数


    def is_valid(row,col):
        for r in range(row):
            if queens[r] == col or abs(row-r) == abs(col - queens[r]):
                return False
        return True

    def backtrack(row):
        if row == n:# 找到一个合法解决方案
            solutions.append(queens.copy())
        else:
            for col in range(n):
                if is_valid(row,col):# 检查当前位置是否合法
                    queens[row] = col
                    backtrack(row+1)
                    queens[row] = -1# 回溯，撤销当前行的选择
    backtrack(0)
    return solutions
# 获取第 b 个皇后串
def get_queen_string(b):
    solutions = solve_n_queen(8)
    if b > len(solutions):
        return None
    queen_string = ''.join(str(col+1) for col in solutions[b-1])
    return queen_string

test_cases = int(input())
for _ in range(test_cases):
    b = int(input())
    queen_string = get_queen_string(b)
    print(queen_string)
#stack means

def queen_stack(n):
    stack = []
    solutions = []
    stack.append((0,[]))

    while stack:
        row,cols = stack.pop() #row, cols = stack.pop() # 从栈中取出当前处理的行数和已放置的皇后位置
        if row == n:
            solutions.append(cols)
        else:
            for col in range(n):
                if is_valid1(row,col,cols):
                    stack.append((row+1,cols+[col]))
    return solutions

def is_valid1(row,col,queens):
    for r in range(row):
        if queens[r] == col or abs(row - r) == abs(col - queens[r]):
            return False
    return True
def get_queen_string1(b):
    solutions = queen_stack(8)
    if b > len(solutions):
        return None
    queen_string = ''.join(str(col+1) for col in solutions[b-1])
    return queen_string
```
02808: 校门外的树
http://cs101.openjudge.cn/practice/02808/
思路：合并
代码
# 
```python
L,M = map(int,input().split())
res = []
for i in range(M):
    start,end = map(int,input().split())
    res.append([start,end])
res.sort(key=lambda x:x[0])
#print(res)
i = 0
j = M-1
while i<j:
    if res[i][1] >= res[i+1][0]:
        end1 = max(res[i][1],res[i+1][1])
        res[i][1] = end1
        res.pop(i+1)
        j-=1
    else:
        i+=1
num1=0
#print(res)
for num in res:
    num1+=(num[1]-num[0]+1)
print(L-num1+1)
```
```python
28190: 奶牛排队
http://cs101.openjudge.cn/practice/28190/

思路： 神奇的单调栈+遍历 每点记录左更大值和右更小值，则左侧可找到最大的更小区间，再遍历j,考虑j右最大区间是否包含i

代码

# 
N = int(input())
heights = [int(input()) for _ in range(N)]
left_bound = [-1]*N
right_bound = [N]*N
stack = []
#单调减小栈，左侧第一个>=h[i]的位置
for i in range(N):
    while stack and heights[stack[-1]]<heights[i]:
        stack.pop()
    if stack:
        left_bound[i] = stack[-1]
    stack.append(i)
stack.clear()
#单调增加栈，右侧第一个<=h[i]的位置
for i in range(N-1,-1,-1):
    while stack and heights[stack[-1]] > heights[i]:
        stack.pop()
    if stack:
        right_bound[i] = stack[-1]
    stack.append(i)
ans = 0
for i in range(N):
    for j in range(left_bound[i]+1,i):
        if right_bound[j] > i:
            ans = max(ans,i-j+1)
            break
#题目，集合，倒排索引
n = int(input())
lis = []#出现文档集合
all_document = set()
for _ in range(n):
    data = list(map(int,input().split()))
    doc_set = set(data[1:])
    lis.append(doc_set)
    all_document.update(doc_set)
# Prepare the not-present sets 未出现文档集合
lis1 = [all_document - doc_set for doc_set in lis]
m = int(input())
for _ in range(m):
    query = list(map(int,input().split()))
    result_set = None
    for num,requirement in enumerate(query):
        if requirement != 0:
            current_set = lis[num] if requirement==1 else lis1[num]
            #取交集
            result_set = current_set if result_set is None else result_set.intersection(current_set)
    if not result_set:
        print("NOT FOUND")
    else:
        print(' '.join(map(str, sorted(result_set))))


print(ans)
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



