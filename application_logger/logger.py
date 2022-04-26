from datetime import datetime
import os

class app_logger:
    def __init__(self):
        pass
    def log(self,file_object,log_message):
        try:
            file_object.write(str(datetime.now())+" "+log_message)
            file_object.write("\n")
            file_object.flush()
            os.fsync(file_object.fileno())
        except Exception as e:
            print("Error in logging message: "+str(e))