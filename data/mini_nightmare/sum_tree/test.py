from code import TreeNode, build_sum_tree, print_tree


def test_shopping_cart():

    # Create a test tree
    root = TreeNode(1)
    node2 = TreeNode(2)
    root.left = node2
    root.right = TreeNode(3)
    node4 = TreeNode(4)
    node5 = TreeNode(5)
    node5.left = node2
    root.left.left = node4
    node4.left = node5

    # Build sum tree and print it
    build_sum_tree(root)
    print_tree(root)
