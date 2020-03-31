import logging


def get_root_logger(log_file=None, log_level=logging.INFO, mode='w'):
    log_level = log_level
    logger = logging.getLogger(__name__.split('.')[0])
    if logger.hasHandlers():
        return logger

    if not log_file:
        print('Must give a file to log!')
    else:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(format=format_str, level=log_level)
        file_handler = logging.FileHandler(log_file, mode=mode)
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def print_log(msg, logger='root', level=logging.INFO):
    if logger is None:
        print(msg)
    elif logger == 'root':
        _logger = get_root_logger()
        _logger.log(level, msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger != 'silent':
        raise TypeError(
            'logger should be either a logging.Logger object, "root", '
            '"silent" or None, but got {}'.format(logger))