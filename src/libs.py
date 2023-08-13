import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import json
import requests
import io
import joblib

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from tensorflow.keras.layers import (
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D,
    Dropout, Conv1D, Dense
)
from tensorflow.keras.callbacks import EarlyStopping