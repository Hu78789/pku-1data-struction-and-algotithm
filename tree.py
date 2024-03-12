class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None
def tree_height(node):
    if node is None:
        return -1
    return max(tree_height(node.left),tree_height(node.right)) + 1
def count_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)
def tree_depth(node):
    if node is None:
        return 0
    left_depth = tree_depth(node.left)
    right_depth = tree_depth(node.right)
    return max(left_depth,right_depth) + 1    
n = int(input())
nodes = [TreeNode() for _ in range(n)]
has_parent = [False] * n
for i in range(n):
    left_index,right_index = map(int,input().split())
    if left_index != -1:
        nodes[i].left = nodes[left_index]
        has_parent[left_index] = True
    if right_index != -1:
        nodes[i].right = nodes[right_index]
        has_parent[right_index] = True

root_index = has_parent.index(False)
root = nodes[root_index]

height = tree_height(root)
leaves = count_leaves(root)
print(f"{height} {leaves}")
