class DUS:#基于多叉树
    def __init__(self,n):
        self.parents = [i for i in range(n)]
        self.rank = [0]*n
    def find(self,x):
        if self.parents[x] != x:
            return self.find(self.parents[x])
        return self.parents[x]
    def union(self,x,y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parents[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parents[rootX] = rootY
            else:
                self.parents[rootY] = rootX
                self.rank[rootX] += 1#rank表多叉树的层次
def solve():
    n,m = map(int,input().split())
    uf = DUS(2*n)#0~n:in ,n+1~2*n:not in
    for i in range(m):
        operation,a,b = input().split()
        a,b = int(a)-1,int(b)-1
        if operation == 'D':
            uf.union(a,b+n)
            uf.union(a+n,b)
        else:
            if uf.find(a) == uf.find(b) or uf.find(a+n) == uf.find(b+n):
                print('In the same gang.')
            elif uf.find(a) == uf.find(b+n) or uf.find(a+n) == uf.find(b):
                print('In different gangs.')
            else:
                print('Not sure yet.')



T = int(input())
for _ in range(T):
    solve()