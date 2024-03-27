import heapq
class TreeNode:
    def __init__(self,char,freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq
def build_huffman_tree(char_freq):
    heap = [TreeNode(char,freq) for char,freq in char_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = TreeNode(None,left.freq+right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap,merged)
    return heap[0]

def encode_huffman_tree(root):
    codes = {}
    def traverse(node,code):
        if node.left is None and node.right is None:
            codes[node.char] = code
        else:
            traverse(node.left,code + '0')
            traverse(node.right,code + '1')
    traverse(root,'')
    return codes

def huffman_encoding(codes,string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root,encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right
        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded




def external_path_length(node,depth=0):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth*node.freq
    return (external_path_length(node.left,depth+1) + external_path_length(node.right,depth+1))

n = int(input().strip())
characters = {}
for _ in range(n):
    char,freq = input().split()
    characters[char] = int(freq)
huffman_tree = build_huffman_tree(characters)
codes = encode_huffman_tree(huffman_tree)
strings = []
while True:
    try:
        line = input().strip()
        strings.append(line)
    except EOFError:
        break
results = []
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree,string))
    else:
        results.append(huffman_encoding(codes,string))
for result in results:
    print(result)
