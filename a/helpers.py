import json
import os

TREE_PATH_EDGE = '-*-'
TREE_PATH_ARRAY = '[]'


def json_file_paths(jsons_path):
    fns = next(os.walk(jsons_path))[2]
    return [os.path.join(jsons_path, fn) for fn in fns]


def new_tpath(tree_path, new_node, list_parent=False):
    if not list_parent:
        if tree_path is None:
            return new_node
        else:
            return '{}{}{}'.format(tree_path, TREE_PATH_EDGE, new_node)
    else:
        if tree_path is None:
            return '{}'.format(TREE_PATH_ARRAY)
        else:
            args = tree_path, TREE_PATH_EDGE, TREE_PATH_ARRAY, TREE_PATH_EDGE, new_node
            return '{}{}{}{}{}'.format(*args)
