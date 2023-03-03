import tensorflow as tf
from tensorflow import keras
from determined.keras import TFKerasTrial, TFKerasTrialContext, InputData



class BTDTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context
        self.data = [] #X_train, X_test, y_train, y_test
        self.data_prep()

    def build_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        model=KNeighborsClassifier(n_neighbors=3)
        return model

    def build_training_data_loader(self):
        return self.data[0],self.data[2].to_numpy()

    def build_validation_data_loader(self):
        return self.data[1],self.data[3].to_numpy()

    def data_prep(self):
        import pandas as pd
        folder_path = self.context.get_hparam("data_location")
        #df=pd.read_csv(f'{folder_path}/diabetes.csv')
        df=pd.read_csv(f'diabetes.csv')
        df_out = df

        X=df_out.drop(columns=['Outcome'])
        y=df_out['Outcome']

        from sklearn.model_selection import train_test_split
        self.data = train_test_split(X,y,test_size=0.2)
        
