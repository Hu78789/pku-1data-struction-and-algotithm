class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0
    def percUp(self,i):
        while i//2>0:
            if self.heapList[i] < self.heapList[i//2]:
                self.heapList[i],self.heapList[i//2] = self.heapList[i//2],self.heapList[i]
            i//=2
    def minChild(self,i):
        if i*2+1 > self.currentSize:
            return i*2
        else:
            if self.heapList[i*2] < self.heapList[i*2+1]:
                return i*2
            else:
                return i*2 + 1
    #水平不分大小
    def percDown(self,i):
        while (i*2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                self.heapList[i],self.heapList[mc] = self.heapList[mc],self.heapList[i]
            i = mc
    def heappush(self,k):
        self.heapList.append(k)
        self.currentSize+=1
        self.percUp(self.currentSize)
    def heappop(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize -= 1
        self.heapList.pop()
        self.percDown(1)
        return retval
    def heapify(self,alist):
        i = len(alist) // 2
        self.heapList = [0] + alist[:]
        while (i>0):
            self.percDown(i)
            i-=1



nums = BinHeap()
def main():
    global nums
    s = input().split()
    if len(s) == 2:
        nums.heappush(int(s[1]))
    else:
        print(nums.heappop())

n = int(input())
for _ in range(n):
    main()

