flag = True
chess = [['' for _ in range(10)] for u in range(10)]
take = [False]*10
n,k,ans = 0,0,0


def dfs(x,y):
    global ans
    global n,k
    if y == k:
        ans += 1
        return
    if x == n:
        return
    for i in range(x,n):
        for j in range(n):
            if chess[i][j] == '#' and not take[j]:
                take[j] = True
                dfs(i+1,y+1)
                take[j] = False
while True:
    n, k = map(int, input().split())
    if n == -1 and k == -1:
        flag = False
        break
    for i in range(n):
        chess[i] = list(input())
    take = [False] * 10
    ans = 0
    dfs(0, 0)
    print(ans)



