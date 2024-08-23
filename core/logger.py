import logging

logging.basicConfig(format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

warnings_dict = []


def warn_once(msg):
    if msg not in warnings_dict:
        logging.warning(msg)
        warnings_dict.append(msg)
