# Assignment #D: May月考

Updated 1654 GMT+8 May 8, 2024

2024 spring, Complied by ==胡景博 药学==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：macOS Ventura 13.4.1 (c)

Python编程环境：Spyder IDE 5.2.2, PyCharm 2023.1.4 (Professional Edition)

C/C++编程环境：Mac terminal vi (version 9.0.1424), g++/gcc (Apple clang version 14.0.3, clang-1403.0.22.14.1)



## 1. 题目

### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：合并



代码

```python
# 
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



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image-6.png)




### 20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/



思路：
利用数学关系


代码

```python
# 
strings = input().strip()
def div_five(strings):
    res=''
    num = 0
    for char in strings:
        num = (num*2 + int(char))%5
        if num == 0:
            res+='1'
        else:
            res+='0'
    return res
print(div_five(strings))
```



代码运行截图 ==（至少包含有"Accepted"）==
![alt text](image-7.png)




### 01258: Agri-Net

http://cs101.openjudge.cn/practice/01258/



思路：prim



代码

```python
# 
import heapq
def prim(graph,start):
    res = []
    visited = set([start])
    edges = [(cost,start,to) for to,cost in graph[start].items()]
    heapq.heapify(edges)
    while edges:
        cost,frm,to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            res.append((frm,to,cost))
        for u,u_cost in graph[to].items():
            if u not in visited:
                visited.add((to,u,u_cost))
                heapq.heappush(edges,(u_cost,to,u))
    return res







def solve():
    N = int(input())
    graph = {i:{} for i in range(N)}

    for i in range(N):
        nums = list(map(int,input().split()))
        for j in range(N):
            if i != j:
                graph[i][j] = nums[j]
    start = 0
    mit = prim(graph,start)
    result = sum(x[2] for x in mit)
    print(result)


while True:
    try:
        solve()
    except EOFError:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-8.png)




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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-9.png)






### 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/



思路：
用堆


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



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-10.png)




### 28190: 奶牛排队

http://cs101.openjudge.cn/practice/28190/



思路：
神奇的单调栈+遍历
每点记录左更大值和右更小值，则左侧可找到最大的更小区间，再遍历j,考虑j右最大区间是否包含i


代码

```python
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
print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==
![alt text](image-11.png)




## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

感觉对算法模型还不太熟练，需要勤加练习



