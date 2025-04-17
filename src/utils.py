import logging
import os

def get_logger(file_path, name="train"):
    log_dir = "/".join(file_path.split("/")[:-1])
    ensure_dir(log_dir)

    logger = logging.getLogger(name)

    hdlr = logging.FileHandler(file_path, mode="a")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    logger.setLevel(logging.INFO)

    return logger

def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system("ln -s {} {}".format(src, target))

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
