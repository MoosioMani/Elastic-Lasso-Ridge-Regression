import pandas as pd

# Drop Nan Columns
def all_nan(df):
    return df.dropna(how='all', axis=0)

# Fill NaN 
def fillna(data):
    data.fillna(value={
        'mileage':data.mileage.mode()[0],
        'engine':data.engine.mode()[0],
        'max_power':data.max_power.mode()[0],
        'torque':data.torque.mode()[0],
        'seats':data.seats.mode()[0],
    },inplace=True)
    return data

# Noisy Data
def noisy_data(data):
    data.loc[data.selling_price.astype('str').str.isalpha(),'selling_price']=data.selling_price.mode()[0]
    return data

# One-Hand Encoder
def one_hand_encoder(data, columns):
    data = pd.get_dummies(data, columns=columns)
    return data

#Label Encoder
from sklearn.preprocessing import LabelEncoder
def label_encoder(data, columns):
    le = LabelEncoder()
    for col in columns:    
        data[col] = le.fit_transform(data[col])
    return data

# Discret 
from sklearn.preprocessing import KBinsDiscretizer
def k_bins_discretizer(data, columns):
    dis = KBinsDiscretizer(n_bins=500, encode='ordinal', strategy='uniform')
    for col in columns:
        data[col] = dis.fit_transform(data[[col]])   
    return data

# Drop Columns
def drop_columns(data, columns):
    for col in columns:
        data.drop(col, axis=1, inplace=True)
    return data

# Outlier Plot 
import plotly.express as px
def outlier_columns_by_plotly(data, columns):
    fig = px.box(data, y=columns)
    fig.show()

# Remove Outlier Rate
def remove_outlier_price(data, min_w, max_w):
    df = pd.DataFrame(data)
    data = df[(df['selling_price'] >= min_w) & (df['selling_price'] <= max_w)]
    return data

def remove_outlier_km(data, min_w, max_w):
    df = pd.DataFrame(data)
    data = df[(df['km_driven'] >= min_w) & (df['km_driven'] <= max_w)]
    return data

def remove_outlier_power(data, min_w, max_w):
    df = pd.DataFrame(data)
    data = df[(df['max_power'] >= min_w) & (df['max_power'] <= max_w)]
    return data