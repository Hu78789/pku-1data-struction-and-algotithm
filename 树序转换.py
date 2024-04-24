def build_tree(preorder,inorder):
    if not preorder:
        return ''
    root = preorder[0]
    root_index = inorder.index(root)

    left_preorder = preorder[1:1+root_index]
    right_preorder = preorder[1+root_index:]

    left_inorder = inorder[:root_index]
    right_inorder = inorder[root_index+1:]

    left_tree = build_tree(left_preorder,left_inorder)
    right_tree = build_tree(right_preorder,right_inorder)
    return left_tree + right_tree + root




while True:
    try:
        preorder,inorder = input().split()
        post_tree = build_tree(preorder, inorder)
        print(post_tree)
    except EOFError:
        break










