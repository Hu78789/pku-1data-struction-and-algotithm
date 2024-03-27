class Node:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1
class AVL:
    def __init__(self):
        self.root = None
    def insert(self,value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value,self.root)

    def _get_height(self,node):
        if not node:
            return 0
        return node.height
    def _get_balance(self,node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _insert(self,value,node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value,node.left)
        else:
            node.right = self._insert(value,node.right)

        node.height = 1 + max(self._get_height(node.left),self._get_height(node.right))
        balance = self._get_balance(node)
        if balance > 1:
            if value < node.left.value:#LL
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:#RR
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        return node
    def _rotate_right(self,y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left),self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left),self._get_height(x.right))
        return x
    def _rotate_left(self,z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y
        def _delete(self,value,node):
        if not node:
            return node
        if value < node.value:
            node.left = self._delete(value,node.left)
        elif value > node.value:
            node.right = self._delete(value,node.right)
        else:
            if not node.left:
                temp = node.right
                node = None
                return temp
            elif not node.right:
                temp = node.left
                node = None
                return temp
            temp = self._min_value_node(node.right)
            node.value = temp.value
            node.right = self._delete(temp.value,node.right)
        if not node:
            return node

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        balance = self._get_balance(node)
        if balance > 1:
            if self._get_balance(node.left) >= 0:
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if self._get_balance(node.right) <= 0:
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node
    def _min_value_node(self,node):
        current = node
        while current.left:
            current = current.left
        return current    
    def preorder(self):
        return self._preorder(self.root)
    def _preorder(self,node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)
n = int(input().strip())
sequence = list(map(int,input().strip().split()))
avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str,avl.preorder())))
