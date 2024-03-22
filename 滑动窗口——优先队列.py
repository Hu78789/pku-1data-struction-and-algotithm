from collections import deque
from typing import List
def maxSlidingWindow(nums:List[int],k:int)->List[int]:
    n = len(nums)
    q = deque()
    #队列从大到小，记录序号
    for i in range(k):
        while q and nums[i] >= nums[q[-1]]:
            q.pop()
        q.append(i)
    ans = [nums[q[0]]]
    for i in range(k,n):
        while q and nums[i] >= nums[q[-1]]:
            q.pop()
        q.append(i)
        while q[0] <= i-k:
            q.popleft()
        ans.append(nums[q[0]])
    return ans
n,k = map(int,input().split())
*nums,=map(int,input().split())
ans = maxSlidingWindow(nums,k)
print(' '.join(map(str,ans)))



