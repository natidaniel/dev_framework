from os.path import join, split, realpath, exists
from os import getcwd, mkdir
import logging
import logging.config
import PIL
import json
import time


def get_log_prefix(log_path=None):
    if log_path is None:
        log_prefix = split(logging.getLogger().handlers[0].baseFilename)[-1]
    else:
        log_prefix = split(log_path)[-1]
    prefix = log_prefix[:log_prefix.rindex('.')]
    return prefix


def record_input_message(name, value):
    if value is not None:
        logging.debug("\t{}:{}".format(name, value))


def get_target_dir(name):
    cwd = getcwd()
    target_dir = join(cwd, name)
    if not exists(target_dir):
        mkdir(target_dir)
    return target_dir


def init_logger():
    path = split(realpath(__file__))[0]
    with open(join(path, 'log.json')) as json_file:
        log_config_dir = json.load(json_file)
        filename = log_config_dir.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M", time.localtime()),".log"])
        log_path = get_target_dir('logs')
        log_config_dir.get('handlers').get('file_handler')['filename'] = join(log_path, filename)
        logging.config.dictConfig(log_config_dir)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)