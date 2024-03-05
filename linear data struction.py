class Stack(object):
    def __init__(self):
        self.items = []
    def is_Empty(self):
        return self.items == []
    def push(self,item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[-1]
    def size(self):
        return len(self.items)
class Deque(object):
    def __init__(self):
        self.items = []
    def is_Empty(self):
        return self.items == []
    def size(self):
        return len(self.items)
    def addFront(self,item):
        self.items.append(item)
    def addRear(self,item):
        self.items.insert(0,item)
    def removeFront(self):
        return self.items.pop()
    def removeRear(self):
        return self.items.pop(0)

#单向链表
class Node1(object):
     def __init__(self,value):
         self.value = value
         self.next = None





class LinkedList(object):
    def __init__(self):
        self.head = None
    def is_Empty(self):
        return self.head is None
    def size(self):
        count = 0
        current = self.head
        while current is not None:
            count+=1
            current = current.next
        return count
    def items(self):
        current = self.head
        while current is not None:
            yield current.value
            current = current.next
    def display(self):
        current = self.head
        while current is not None:
            print(current.value)
            current = current.next
    def insert_head(self,value):
        new_node = Node1(value)
        new_node.next = self.head
        self.head = new_node
    def append(self,value):
        new_node = Node1(value)
        if self.is_Empty():
            self.head = new_node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = new_node
    def insert_index(self,index,value):
        if index <= 0:
            self.insert_head(value)
        elif index > (self.size()-1):
            self.append(value)
        else:
            new_node = Node1(value)
            current = self.head
            for _ in range(index-1):
                current = current.next
            new_node.next = current.next
            current.next = new_node
    def remove_index(self,index):

        if index <= 0:
            return
        elif index > (self.size()-1):
            return
        else:

            current = self.head
            pre=Node1(None)
            pre.next = current
            for _ in range(index-1):
                pre = pre.next
                current = current.next
            pre.next = current.next
            current.next = None

    def remove(self, value):
        current = self._head
        pre = None
        while current is not None:
            if current.value == value:
                if not pre:
                    self._head = current.next
                else:
                    pre.next = current.next
                return True
            else:
                pre = current
                current = current.next
    def find(self,value):
        return value in self.items()
if __name__ == "__main__":
    link_list = LinkedList()
    node1 = Node1(1)
    node2 = Node1(2)
    link_list._head = node1
    node1.next = node2
    print(link_list._head.value)
    print(link_list._head.next.value)
    print(link_list.is_empty())
    print(link_list.length())
    for i in range(7):
        link_list.append(i)
    link_list.insert_head(10)
    print(link_list.length())
    print(list(link_list.items()))
    link_list.insert_index(3, 20)
    print(list(link_list.items()))
    print(link_list.find(10))
    link_list.remove(2)
    print(list(link_list.items()))
#循环列表
# 定义节点
class Node():
    # 初始化
    def __init__(self, value):
        self.value = value
        self.next = None


# 定义链表(单向链表)
class CircleLinkList():
    # 初始化
    def __init__(self):
        self.head = None

    # 判断链表是否为空
    def is_empty(self):
        return self.head is None

    # 链表长度
    def length(self):
        if self.is_empty():
            return 0
        count = 1
        current = self.head
        while current.next != self.head:
            count += 1
            current = current.next
        return count

    # 遍历链表
    def items(self):
        current = self.head
        while current.next != self.head:
            yield current.value
            current = current.next
        yield current.value

    # 向链表头部添加元素
    def insert_head(self, value):
        new_node = Node(value)
        if self.head is not None:
            new_node.next = self.head
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = new_node
        else:
            self.head = new_node
            new_node.next = self.head
        self.head = new_node

    # 尾部添加元素
    def append(self, value):
        new_node = Node(value)
        if self.head is not None:
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = new_node
            new_node.next = self.head
        else:
            self.head = new_node
            new_node.next = self.head

    # 指定位置插入元素
    def insert(self, index, value):
        if index <= 0:  # 指定位置小于等于0，头部添加
            self.insert_head(value)
        elif index > self.length()-1:
            self.append(value)
        else:
            new_node = Node(value)
            current = self.head
            for _ in range(index-1):
                current = current.next
            new_node.next = current.next
            current.next = new_node

    # 删除节点
    def remove(self, value):
        # 若链表为空
        if self.is_empty():
            return
        current = self.head
        pre = Node
        # 如果第一个元素为需要删除的元素
        if current.value == value:
            # 如果链表不止一个元素
            if current.next != self.head:
                while current.next != self.head:
                    current = current.next
                current.next = self.head.next
                self.head = self.head.next
            # 如果只有一个元素
            else:
                self.head = None
        # 如果删除的是链表中间的元素
        else:
            pre = self.head
            while current.next != self.head:
                if current.value == value:
                    pre.next = current.next
                    return True
                else:
                    pre = current
                    current = current.next
        # 如果删除的为结尾的元素
        if current.value == value:
            pre.next = self.head
            return True

    # 查找元素是否存在
    def find(self, value):
        return value in self.items()


if __name__ == "__main__":
    circle_link = CircleLinkList()
    print(circle_link.is_empty())
    print(circle_link.length())
    circle_link.insert_head(10)
    print(circle_link.is_empty())
    circle_link.insert_head(20)
    circle_link.insert_head(1)
    circle_link.insert_head(0)
    print(list(circle_link.items()))
    for i in range(2, 9):
        circle_link.append(i)
    print(list(circle_link.items()))
    print(circle_link.find(10))
    circle_link.insert(5, 30)
    print(list(circle_link.items()))
    circle_link.remove(30)
    print(list(circle_link.items()))
    circle_link.remove(0)
    print(list(circle_link.items()))
    circle_link.remove(8)
    print(list(circle_link.items()))
    circle_link.remove(80)
    print(list(circle_link.items()))
#双向列表
# 定义节点
class Node2():
    # 初始化
    def __init__(self, value):
        self.value = value
        self.next = None
        self.pre = None


# 定义链表(单向链表)
class DoubleLinkedList():
    # 初始化
    def __init__(self):
        self._head = None

    # 判断链表是否为空
    def is_empty(self):
        return self._head is None

    # 链表长度
    def length(self):
        count = 0
        current = self._head
        while current is not None:
            count = count + 1
            current = current.next
        return count

    # 遍历链表
    def items(self):
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    # 向链表头部添加元素
    def insert_head(self, value):
        new_node = Node2(value)
        # 链表为空时
        if self._head is None:
            # 头部结点指针修改为新结点
            self._head = new_node
        else:
            # 新节点指向原来的头部节点
            new_node.next = self._head
            # 原来头部节点pre指向新节点
            self._head.pre = new_node
            # head指向新节点
            self._head = new_node

    # 尾部添加元素
    def append(self, value):
        new_node = Node2(value)
        if self._head is None:
            # 头部结点指针修改为新结点
            self._head = new_node
        else:
            current = self._head
            while current.next is not None:
                current = current.next
            current.next = new_node
            new_node.pre = current

    # 指定位置插入元素
    def insert(self, index, value):
        if index <= 0:  # 指定位置小于等于0，头部添加
            self.insert_head(value)
        elif index > self.length()-1:
            self.append(value)
        else:
            current = self._head
            new_node = Node2(value)
            for _ in range(index-1):
                current = current.next
            # 新节点的前一个节点指向当前节点的上一个节点
            new_node.pre = current.pre
            # 新节点的下一个节点指向当前节点
            new_node.next = current
            # 当前节点的上一个节点指向新节点
            current.pre.next = new_node
            # 当前结点的向上指针指向新结点
            current.pre = new_node

    # 删除节点
    def remove(self, value):
        if self.is_empty():
            return
        current = self._head
        # 删除的元素为第一个元素
        if current.value == value:
            # 链表中只有一个元素
            if current.next is None:
                self._head = None
                return True
            else:
                self._head = current.next
                current.next.pre = None
                return True
        while current.next is not None:
            if current.value == value:
                current.pre.next = current.next
                current.next.pre = current.pre
                return True
            current = current.next
        # 删除元素在最后一个
        if current.value == value:
            current.pre.next = None
            return True

    # 查找元素是否存在
    def find(self, value):
        return value in self.items()


if __name__ == "__main__":
    link_list = DoubleLinkedList()
    print(link_list.is_empty())
    print(link_list.length())
    link_list.insert_head(10)
    print(link_list.is_empty())
    link_list.insert_head(20)
    link_list.insert_head(1)
    link_list.insert_head(0)
    print(list(link_list.items()))
    for i in range(2, 9):
        link_list.append(i)
    print(list(link_list.items()))
    print(link_list.find(10))
    link_list.remove(0)
    print(list(link_list.items()))
    link_list.remove(3)
    print(list(link_list.items()))
    link_list.remove(8)
    print(list(link_list.items()))