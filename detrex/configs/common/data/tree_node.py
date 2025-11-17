class TreeNode:
    def __init__(self, id, label, bbox, category):
        self.id = id
        self.label = label
        self.bbox = bbox
        self.category = category
        # List[TreeNode]
        self.children = []
