import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('../../train.csv', index_col='Id');
train_data['MSSubClass'].value_counts().plot(kind='barh',rot = 0)
plt.show()