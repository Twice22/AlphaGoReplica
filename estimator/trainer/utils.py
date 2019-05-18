import os
import re

from config import *
from glob import glob

_CHECKPOINT_RE = re.compile(r"([\w\d]*)\.ckpt-([0-9]*)")


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


def get_komi():
    if not FLAGS.is_go:
        return 0

    if 14 <= FLAGS.n_rows <= 19:
        return 7.5
    elif 9 <= FLAGS.n_rows <= 13:
        return 5.5
    return 0
