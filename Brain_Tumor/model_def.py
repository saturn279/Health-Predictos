import tensorflow as tf
from tensorflow import keras

from determined.keras import TFKerasTrial, TFKerasTrialContext, InputData

import data


class BTDTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext) -> None:
        self.context = context
        self.prep_data = [] #X_train, X_test, y_train, y_test
        self.data_prep()

    def build_model(self):

        import os
        import warnings
        import itertools
        import cv2
        import seaborn as sns
        import pandas as pd
        import numpy  as np
        from PIL import Image
        from sklearn.utils import class_weight
        from sklearn.metrics import confusion_matrix, classification_report
        from collections import Counter

        import tensorflow as tf

        import tensorflow_addons as tfa
        import visualkeras
        import plotly.express as px
        import matplotlib.pyplot as plt
        from sklearn.metrics import multilabel_confusion_matrix

        from tensorflow.keras.preprocessing.image import load_img
        from tensorflow.keras.utils import plot_model
        from tensorflow.keras import layers
        from tensorflow.keras import regularizers
        from sklearn.model_selection   import train_test_split
        from keras.preprocessing.image import ImageDataGenerator

        # General parameters
        import numpy as np
        epochs = 15
        pic_size = 240
        np.random.seed(42)
        tf.random.set_seed(42)
        model = tf.keras.Sequential([
    
                tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2), activation="relu", padding="valid",input_shape=(pic_size,pic_size,3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2), activation="relu", padding="valid"),
                tf.keras.layers.MaxPooling2D((2, 2)),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=64, activation='relu', 
                                      kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3), 
                                      bias_regularizer=regularizers.L2(1e-2),
                                      activity_regularizer=regularizers.L2(1e-3)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=1, activation='sigmoid'),
            ])
        # Wrap the model.
        model = self.context.wrap_model(model)

        # Create and wrap the optimizer.
        optimizer = tf.keras.optimizers.Adam()
        optimizer = self.context.wrap_optimizer(optimizer)
        
        model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
        return model

    def build_training_data_loader(self):
        return self.data[0],self.data[2]

    def build_validation_data_loader(self):
        return self.data[1],self.data[3]
    



# for image_name in non_tumorous_dataset_list:
#     print(tumorous_dataset_path + image_name)
#     image=cv2.imread(non_tumorous_dataset_path + image_name)
#     image=Image.fromarray(image,'RGB')

    def data_prep(self):
        import os
        import warnings
        import itertools
        import cv2
        import seaborn as sns
        import pandas as pd
        import numpy  as np
        from PIL import Image
        from sklearn.utils import class_weight
        from sklearn.metrics import confusion_matrix, classification_report
        from collections import Counter

        import tensorflow as tf

        import tensorflow_addons as tfa
        import visualkeras
        import plotly.express as px
        import matplotlib.pyplot as plt
        from sklearn.metrics import multilabel_confusion_matrix

        from tensorflow.keras.preprocessing.image import load_img
        from tensorflow.keras.utils import plot_model
        from tensorflow.keras import layers
        from tensorflow.keras import regularizers
        from sklearn.model_selection   import train_test_split
        from keras.preprocessing.image import ImageDataGenerator
        folder_path = "input_dataset"
        non_tumorous_dataset_list = os.listdir("Dataset/no/")
        non_tumorous_dataset_path= "Dataset/no/"
        tumorous_dataset_list = os.listdir("Dataset/yes/")
        tumorous_dataset_path = "Dataset/yes/"
        non_tumorous_dataset_array=[]
        tumorous_dataset_array=[]
        for image_name in non_tumorous_dataset_list:
            try:
                print(non_tumorous_dataset_path+ image_name)
                image=cv2.imread(non_tumorous_dataset_path+ image_name)
                image=Image.fromarray(image,'RGB')
                image=image.resize((240,240))
                non_tumorous_dataset_array.append(np.array(image))
                tumorous_dataset_array.append(0)
            except AttributeError:
                print(non_tumorous_dataset_path+ image_name)

        for image_name in tumorous_dataset_list:
            try:
                image=cv2.imread(tumorous_dataset_path + image_name)
                image=Image.fromarray(image,'RGB')
                image=image.resize((240,240))
                non_tumorous_dataset_array.append(np.array(image))
                tumorous_dataset_array.append(1)
            except AttributeError:
                print(non_tumorous_dataset_path+ image_name)
        non_tumorous_dataset_array = np.array(non_tumorous_dataset_array)
        tumorous_dataset_array = np.array(tumorous_dataset_array)
        self.data = train_test_split(non_tumorous_dataset_array, tumorous_dataset_array, test_size=0.2, shuffle=True, random_state=42)