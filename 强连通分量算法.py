#Kosaraju算法的Python实现,不一定双向强联通
'''
Kosaraju算法是一种用于在有向图中寻找强连通分量（Strong Connected Components，SCC）的算法。它基于深度优先搜索（DFS）和图的转置操作。

Kosaraju算法的核心思想就是两次深度优先搜索（DFS）。

第一次DFS：在第一次DFS中，我们对图进行标准的深度优先搜索，但是在此过程中，我们记录下顶点完成搜索的顺序。这一步的目的是为了找出每个顶点的完成时间（即结束时间）。

反向图：接下来，我们对原图取反，即将所有的边方向反转，得到反向图。

第二次DFS：在第二次DFS中，我们按照第一步中记录的顶点完成时间的逆序，对反向图进行DFS。这样，我们将找出反向图中的强连通分量。

Kosaraju算法的关键在于第二次DFS的顺序，它保证了在DFS的过程中，我们能够优先访问到整个图中的强连通分量。因此，Kosaraju算法的时间复杂度为O（V + E），其中V是顶点数，E是边数。
'''
graph = [[1], [2, 4], [3, 5], [0, 6], [5], [4], [7], [5, 6]]#节点是索引
def dfs1(graph,node,visited,stack):
    #stack:表示遍历时间
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs1(graph,neighbor,visited,stack)
    stack.append(node)
def dfs2(graph,node,visited,component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs2(graph,neighbor,visited,component)




def kosaraju(graph):
    stack = []
    visited = [False] * len(graph)
    for node in range(len(graph)):
        if not visited[node]:
            dfs1(graph,node,visited,stack)
    #反向图
    transposed_graph = [[] for _ in range(len(graph))]
    for node in range(len(graph)):
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)
    visited = [False] * len(graph)
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(transposed_graph,node,visited,scc)
            sccs.append(scc)
    return sccs


sccs = kosaraju(graph)
print("Strongly Connected Components:")
for scc in sccs:
    print(scc)


