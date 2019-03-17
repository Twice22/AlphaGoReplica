import os
import re
from glob import glob

_CHECKPOINT_RE = re.compile(r"([\w\d]*)\.ckpt-([0-9]*)")


def create_configuration(mapping, filename="config.py"):
    with open(filename, 'w') as f:
        for key, value in mapping.items():
            if isinstance(value, str):
                value = "\"" + value + "\""
            f.write("%s = %s\n" % (key, value))


def checkpoints_already_exist(model_path):
    return len(glob(os.path.join(model_path, "*"))) > 0


def latest_checkpoint(ckpt_dir):
    """
    Return the path to the latest checkpoint
    Args:
        ckpt_dir (str): path to the directory with multiple checkpoints
    Returns:
        str: path to the latest checkpoint
    Raises:
        RuntimeError: If no checkpoint could have been found in the `ckpt_dir` directory
    """
    checkpoints = glob(os.path.join(ckpt_dir, "*"))
    list_checkpoints = []

    for checkpoint in checkpoints:
        m = _CHECKPOINT_RE.search(checkpoint)
        if m:
            list_checkpoints.append((m.group(2), m.group(0)))

    if list_checkpoints:
        last_checkpoint = sorted(list_checkpoints)[-1][1]
        return os.path.join(ckpt_dir, last_checkpoint)
    else:
        raise RuntimeError("No checkpoint could have been found in %s" % ckpt_dir)
