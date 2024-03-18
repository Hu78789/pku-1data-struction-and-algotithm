#中缀转后缀
def infixToPostfix(infixexpr):
    prec = {}
    prec['*'] = 3
    prec['/'] = 3
    prec['+'] = 2
    prec['-'] = 2
    prec['('] = 1
    stack = []
    postfixList = []
    tokenList = infixexpr.split()
    for token in tokenList:
        if token.isnumeric():
            postfixList.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            top = stack.pop()
            while top != '(':
                postfixList.append(top)
                top = stack.pop()
        else:
            while (not stack) and \
                    (prec[stack[-1]]) >= prec[token]:
                postfixList.append(stack.pop())
            stack.append(token)
    while not stack:
        postfixList.append(stack.pop())
    return " ".join(postfixList)
post = '2 3 +'
def postfixEval(postfixExpr):
    stack = []
    tokenList = postfixExpr.split()
    for token in tokenList:
        if token.isnumeric():
            stack.append(int(token))
        else:
            op2 = stack.pop()
            op1 = stack.pop()
            result = eval(str(op1) + token + str(op2))
            stack.append(result)
    return stack.pop()
print(postfixEval(post))