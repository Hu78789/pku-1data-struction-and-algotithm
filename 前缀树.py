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












