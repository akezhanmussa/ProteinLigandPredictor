'''
    custom logger object,
    contains the FileHandler
    and StreamHandler by default
'''

import logging 

class Logger():
    """The class wrapper of the logger object
    
    :param path: the path to the log file 
    :type path: str
    :param name: the name of the logger object
    :type name: logging.Logger 
    :param level: the level of logger
    :type level: int
    :param warning_to_file: 'True' if the warnings are wished to be recorded, 'False' otherwise
    :type warining_to_file: bool, optional 
    :param override_log: 'True' if the current log file is wished to be rewritten, 'False' otherwise
    :type override_log: bool, optional
    """

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
        """Logs the message with DEBUG level
        
        :param msg: message
        :type msg: str
        """
        self.logger.debug(msg)
    
    def info(self, msg):
        """Logs the message with INFO level
        
        :param msg: message
        :type msg: str
        """
        self.logger.info(msg)

    def warning(self, msg):
        """Logs the message with WARNING level
        
        :param msg: message
        :type msg: str
        """
        self.logger.warning(msg)
    
    def error(self, msg):
        """Logs the message with error level
        
        :param msg: [description]
        :type msg: [type]
        """
        self.logger.error(msg)

    




    







        


