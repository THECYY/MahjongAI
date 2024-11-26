from py_utils import *

class Logger:
    
    def __init__(self, file=None, kmode=False):
        self.kmode = kmode
        self.file = file
        self.enable = True
        self.clear_logger()

    def stop(self):
        self.enable = False
    
    def start(self):
        self.enable = True
    
    def clear_logger(self):
        self.message_list = []
        self.importent_list = []
        self._console_len = 0
        self._log_len = 0

    def trace(self, message: str):
        if not self.enable:
            return 
        self._log("trace", message)
    
    def warning(self, message: str):
        if not self.enable:
            return 
        self._log("warning", message)
    
    def error(self, message: str):
        if not self.enable:
            return 
        self._log("error", message)

    def importent(self, message: str):
        if not self.enable:
            return 
        self._log("imp", message)

    def _log(self, title, message: str):
        system_time = system_time_str()
        title = expand_str(title, 7)
        message = "[ {} | {} ] {}".format(system_time, title, message)
        self.message_list.append(message)
        if self.kmode and self.file != None:
            self.log_to_file(self.file, False)
        if title[-3:] == "imp":
            self.importent_list.append(message)

    def print_to_console(self, force_all=False):
        if not self.enable:
            return
        start = self._console_len if not force_all else 0
        for i in range(start, len(self.message_list)):
            print(self.message_list[i])
        self._console_len = len(self.message_list)

    def log_to_file(self, file, force_all=False):
        if not self.enable:
            return
        with open(file, "a") as f:
            start = self._log_len if not force_all else 0
            for i in range(start, len(self.message_list)):
                print(self.message_list[i], file=f)
            self._log_len = len(self.message_list)

    def importent_log_to_file(self, file):
        if not self.enable:
            return
        with open(file, "a") as f:
            for message in self.importent_list:
                print(message, file=f)


    