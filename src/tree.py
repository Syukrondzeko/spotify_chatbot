import os


def print_directory_tree(root_dir, prefix=""):
    items = os.listdir(root_dir)
    for index, item in enumerate(items):
        path = os.path.join(root_dir, item)
        is_last = index == len(items) - 1
        print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
        if os.path.isdir(path):
            print_directory_tree(path, prefix + ("    " if is_last else "│   "))


print_directory_tree(".")
