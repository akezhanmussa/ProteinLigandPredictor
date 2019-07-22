'''
    custom logger object,
    contains the FileHandler
    and StreamHandler by default
'''

import logging 

class Logger():

    def __init__(self, path, name = 'logger', level = logging.DEBUG, warning_to_file = False, override_log = False):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        log_format = logging.Formatter(fmt = "%(asctime)s:%(message)s", datefmt = "%m/%d/%Y")
        log_format_console = logging.Formatter(fmt = "%(asctime)s:%(levelname)s:%(message)s", datefmt = "%m/%d/%Y")
        
        name_log ="{0}/{1}.log".format(path, name)

        file_handler = logging.FileHandler(name_log, mode = 'w' if override_log else 'a')
        file_handler.setFormatter(log_format)
        file_handler.setLevel(logging.INFO if not warning_to_file else logging.WARNING)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format_console)
        console_handler.setLevel(logging.INFO)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, msg):
        self.logger.debug(msg)
    
    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)

    




    







        


