import json


def load_config(path):
    with open(path, "r") as f:
        config = json.loads(f.read())
    return config