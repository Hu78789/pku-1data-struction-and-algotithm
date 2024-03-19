
class dir:
    def __init__(self,name):
        self.name = name
        self.dirs = []
        self.files = []
def printStruction(root,indent = 0):
    pre = '|     '*indent
    print(pre + root.name)
    for dir in root.dirs:
        printStruction(dir,indent+1)
    for file in root.files:
        print(pre + file)
temp = []
datas = []
dataset = 1
while True:
    line = input()
    if line == '#':
        break
    if line == '*':
        datas.append(temp)
        temp = []
    else:
        temp.append(line)
for data in datas:
    print(f'DATA SET {dataset}:')
    root = dir('ROOT')
    stack = [root]
    for line in data:
        if line[0] == 'd':
            stack[-1].dirs.append(dir(line))
            stack.append(dir(line))
        elif line[0] == 'f':
            stack[-1].files.append(line)
        else:
            stack.pop()
    printStruction(root)
    if dataset < len(datas):
        print()
    dataset += 1


