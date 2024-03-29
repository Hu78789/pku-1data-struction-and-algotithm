class TreeNode:
    def __init__(self,value):
        self.value = value
        self.left = None
        self.right = None
def build_tree(s):
    stack = []
    i = 0
    while i < len(s):
        if s[i].isdigit() or s[i] == '-':
            j=i
            while j < len(s) and (s[j].isdigit() or s[j] == '-'):
                j+=1
            num = int(s[i:j])
            node = TreeNode(num)
            if stack:
                parent = stack[-1]
                if parent.left is None:
                    parent.left = node
                else:
                    parent.right = node
            stack.append(node)
            i = j
        elif s[i] == '(':

            i+=1
        elif s[i] == ')' and s[i-1] != '(' and len(stack)>1:#?
            stack.pop()
            i+=1
        else:
            i+=1
    return stack[0] if len(stack) > 0 else None
def has_path_sum(root,target):
    if root is None:
        return False
    if root.left is None and root.right is None:
        return root.value == target
    left_exist = has_path_sum(root.left,target - root.value)
    right_exist = has_path_sum(root.right,target - root.value)
    return left_exist or right_exist
test_cases = []
while True:
    try:
        line = input().strip()
        if line == '':
            break
        else:
            if line[0].isnumeric():
                test_cases.append(line)
            else:
                test_cases[-1]+=line
    except EOFError:
        break

# 处理每个测试用例并输出结果
for test_case in test_cases:
    I, T = test_case.split(' ', 1)
    target_sum = int(I)
    tree = build_tree(T)
    result = "yes" if has_path_sum(tree, target_sum) else "no"
    print(result)




