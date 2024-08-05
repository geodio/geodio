import logging

logging.basicConfig(format="{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                    level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)