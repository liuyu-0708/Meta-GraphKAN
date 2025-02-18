import logging
import sys


class StdLogger():
    def __init__(self, file_path = "", level=logging.INFO):
        formatter = logging.Formatter(fmt='%(asctime)s %(message)s',
                                      datefmt="%H:%M:%S")
        self.logger = logging.getLogger(__file__)
        self.logger.setLevel(level)

        if file_path:
            file_hander = logging.FileHandler(file_path)
            file_hander.setFormatter(formatter)
            self.logger.addHandler(file_hander)

        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)


Logger = StdLogger("",
                   level=logging.INFO).logger