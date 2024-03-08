#递归回溯
def solve_n_queen(n):
    solutions = []
    queens = [-1] * n    #queens = [-1] * n  # 存储每一行皇后所在的列数


    def is_valid(row,col):
        for r in range(row):
            if queens[r] == col or abs(row-r) == abs(col - queens[r]):
                return False
        return True

    def backtrack(row):
        if row == n:# 找到一个合法解决方案
            solutions.append(queens.copy())
        else:
            for col in range(n):
                if is_valid(row,col):# 检查当前位置是否合法
                    queens[row] = col
                    backtrack(row+1)
                    queens[row] = -1# 回溯，撤销当前行的选择
    backtrack(0)
    return solutions
# 获取第 b 个皇后串
def get_queen_string(b):
    solutions = solve_n_queen(8)
    if b > len(solutions):
        return None
    queen_string = ''.join(str(col+1) for col in solutions[b-1])
    return queen_string

test_cases = int(input())
for _ in range(test_cases):
    b = int(input())
    queen_string = get_queen_string(b)
    print(queen_string)
#stack means

def queen_stack(n):
    stack = []
    solutions = []
    stack.append((0,[]))

    while stack:
        row,cols = stack.pop() #row, cols = stack.pop() # 从栈中取出当前处理的行数和已放置的皇后位置
        if row == n:
            solutions.append(cols)
        else:
            for col in range(n):
                if is_valid1(row,col,cols):
                    stack.append((row+1,cols+[col]))
    return solutions

def is_valid1(row,col,queens):
    for r in range(row):
        if queens[r] == col or abs(row - r) == abs(col - queens[r]):
            return False
    return True
def get_queen_string1(b):
    solutions = queen_stack(8)
    if b > len(solutions):
        return None
    queen_string = ''.join(str(col+1) for col in solutions[b-1])
    return queen_string

