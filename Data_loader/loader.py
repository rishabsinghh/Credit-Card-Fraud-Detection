import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self,logger_object,file_object):
        self.logger_object=logger_object
        self.file_object=file_object
    def read_from_files(self,DIR_INPUT, BEGIN_DATE, END_DATE):
        files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f>=BEGIN_DATE+'.pkl' and f<=END_DATE+'.pkl']

        frames = []
        for f in files:
            df = pd.read_pickle(f)
            frames.append(df)
            del df
        df_final = pd.concat(frames)
        
        df_final=df_final.sort_values('TRANSACTION_ID')
        df_final.reset_index(drop=True,inplace=True)
        #  Note: -1 are missing values for real world data 
        df_final=df_final.replace([-1],0)
    
    def load_data(self,DIR_INPUT, BEGIN_DATE, END_DATE):
        try:
            self.logger_object.log(self.file_object,"Entered into load_data method of Data_loader class")
            transactions_df=self.read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE)
            self.logger_object.log(self.file_object,"{0} transactions loaded, containing {1} fraudulent transactions".format(len(transactions_df),transactions_df.TX_FRAUD.sum()))
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,"Error in loading data: "+str(e))
            return None