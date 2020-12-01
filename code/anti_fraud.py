from datetime import datetime
start_time = datetime.now()

import numpy as np
np.random.seed(1234)

# prevent warnings from distracting the reader
import warnings
warnings.filterwarnings('ignore')

# load the dataset into the notebook kernel
from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file

import pandas as pd
ori_dataset = pd.read_csv(filename)

# remove the "ground-truth" label
if 'label' in ori_dataset:
    label = ori_dataset.pop('label')

categ_cols = ori_dataset.select_dtypes([np.object]).columns

# encode categorical attributes into a binary one-hot encoded representation 
ori_dataset_categ_transformed = pd.get_dummies(ori_dataset[categ_cols])

numeric_cols = ori_dataset.select_dtypes([np.int64, np.float64, np.uint64]).columns

# add a small epsilon to eliminate zero values from data for log scaling
numeric_attr = ori_dataset[numeric_cols] + 1e-7
numeric_attr = numeric_attr.apply(np.log)

# normalize all numeric attributes to the range [0,1]
ori_dataset_numeric_attr = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())

# merge categorical and numeric subsets
ori_subset_transformed = pd.concat([ori_dataset_categ_transformed, ori_dataset_numeric_attr], axis = 1)

df = ori_subset_transformed.copy()

from sklearn.preprocessing import MinMaxScaler
ss = MinMaxScaler()
df_ss = ss.fit_transform(df)

# latent space dimension
encoding_dim = 2

# input placeholder
from keras.layers import Input
input_data = Input(shape = (df_ss.shape[1],))

# encoded input
from keras.layers import Dense
from keras import regularizers
encoded = Dense(512, activation = 'relu', activity_regularizer = regularizers.l1(10e-5) ) (input_data)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)
encoded = Dense(4, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# decoded input
decoded = Dense(4, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(decoded)
decoded = Dense(32, activation='relu')(decoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(df_ss.shape[1], activation='sigmoid')(decoded)

# build autoencoder model
from keras.models import Model
autoencoder = Model (input_data, decoded)

# build encoder for autoencoder model
encoder = Model (input_data, encoded)

autoencoder.compile (optimizer = 'adam', loss = 'binary_crossentropy')

epochs = 100
batch_size = 64
autoencoder_history = autoencoder.fit(
    df_ss, 
    df_ss, 
    epochs = epochs,
    batch_size = batch_size,
    shuffle = False,
    verbose = 1,
    validation_split = (1 / 3),
).history

anomaly_ratio = 0.0002

head = int(anomaly_ratio * df_ss.shape[0])

threshold = 0.019

X_pred = autoencoder.predict(df_ss)
X_pred = pd.DataFrame(X_pred, columns = df.columns)
X_pred.index = df.index

autoencoder_scored = pd.DataFrame(index = df.index)
autoencoder_scored['anomaly_score'] = np.mean(np.abs(X_pred-df_ss), axis=1)
autoencoder_scored = autoencoder_scored.sort_values('anomaly_score', ascending=False).head(head)
autoencoder_scored['threshold'] = threshold
autoencoder_scored['pred_anomaly'] = (autoencoder_scored.anomaly_score >= autoencoder_scored.threshold)

df_results = autoencoder_scored[(autoencoder_scored.pred_anomaly == True)]['pred_anomaly']
df_results = pd.concat([ori_dataset.iloc[df_results.index]], axis = 1)

html_string = '''
<html>
  <head><title>Fraud Scan Results</title></head>
  <link rel="stylesheet" type="text/css" href="../code/df_style.css"/>
  <body>
    <h1 align = 'center'>Fraud Scan Results</h1>
    <p align = 'center'>Review {n_anomalies} detected issues</p>
    <p align = 'center'>{table}</p>
    <p align = 'center'>Time Taken: {time_taken}</p>
  </body>
</html>.
'''

# OUTPUT AN HTML FILE
with open('../data/results.html', 'w') as f:
    f.write(
        html_string.format(
            n_anomalies = \
            df_results.shape[0],
            table=df_results.to_html(classes='mystyle'), 
            time_taken=(datetime.now() - start_time)
        )
    )
    