from collections import deque,defaultdict
graph = {
    'cup_milk': ['mix_ingredients'],
    'mix_ingredients': ['pour_batter', 'heat_syrup'],
    'pour_batter': ['turn_pancake'],
    'turn_pancake': ['eat_pancake'],
    'heat_syrup': ['eat_pancake'],
    'heat_griddle': ['pour_batter'],
    'tbl_oil': ['mix_ingredients'],
    'egg': ['mix_ingredients'],
    'eat_pancake': []
}
def topological_sort(graph):
    in_degree = defaultdict(int)
    result = []
    queue = deque()

    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    for u in graph:
        if in_degree[u] == 0:
            queue.append(u)
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph[u]:
            in_degree[v]-=1
            if in_degree[v] == 0:
                queue.append(v)
    if len(result) == len(graph):
        return result
    else:
        return None
sorted_vertices = topological_sort(graph)
if sorted_vertices:
    print(sorted_vertices)
else:
    print('cycle wrong!')

