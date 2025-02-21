class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.sum = 0  # Will store sum of all nodes below this one

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

def build_sum_tree(root):
    if not root:
        return 0
    
    left_sum = build_sum_tree(root.left)
    right_sum = build_sum_tree(root.right)
    
    root.sum = left_sum + right_sum + root.value
    return root.sum


def print_tree(node, level=0):
    output = []
    if node:
        output.append(f"Value: {node.value}, Sum: {node.sum}")
        output += print_tree(node.left, level + 1)
        output += print_tree(node.right, level + 1)
    return output
