from sum_tree_code import TreeNode, build_sum_tree, print_tree


def test_shopping_cart():

    # Create a test tree
    root = TreeNode(1)
    node2 = TreeNode(2)
    root.set_left(node2)
    root.set_right(TreeNode(3))
    node4 = TreeNode(4)
    node5 = TreeNode(5)
    node5.set_left(node2)
    root.left.set_left(node4)
    node4.set_left(node5)

    # Build sum tree and print it
    build_sum_tree(root)
    output = print_tree(root)
    
    assert output == ['Value: 1, Sum: 15', 'Value: 2, Sum: 11', 'Value: 4, Sum: 9', 'Value: 5, Sum: 5', 'Value: 3, Sum: 3']
