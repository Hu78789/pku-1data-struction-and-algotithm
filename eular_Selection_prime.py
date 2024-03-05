def eular_select(n):
    prime = [0 for _ in range(n+1)]
    common = []
    for i in range(2,n):

        if prime[i] == 0:
            common.append(i)

        for j in common:
            if i*j > n:
                break
            prime[i*j] = 1
            if i%j == 0:
                break
    return prime
s = eular_select(1000000)
n = int(input())
for i in map(int,input().split()):
    if i < 4:
        print('NO')
        continue
    elif int(i**0.5)**2 != i:
        print('NO')
        continue
    elif s[int(i**0.5)] == 0:

        print('YES')
        continue
    else:
        print('NO')