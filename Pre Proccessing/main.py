from extraction import *
from transform import *
from load import *

def seperate(data):
    print(data)
    print(80*'*')

data = extract_from_csv('./used_cars.csv')
seperate(data)

data = data.drop_duplicates(inplace=False) 

data['engine'] = data['engine'].str.extract(r'(\d+)').astype(float)
data['mileage'] = data['mileage'].str.extract(r'(\d+)').astype(float)
data['max_power'] = data['max_power'].str.extract(r'(\d+)').astype(float)
data['torque'] = data['torque'].str.extract(r'(\d+)').astype(float)

data = fillna(data)

data = noisy_data(data)

data = one_hand_encoder(data, ['transmission'])

data = label_encoder(data, ['fuel', 'owner', 'seller_type'])

data = drop_columns(data, ['name'])

data = k_bins_discretizer(data, ['km_driven'])
seperate(data)

data = remove_outlier_price(data, 0, 6000000)

data = remove_outlier_km(data, 0, 200)

data = remove_outlier_power(data, 0, 210)
seperate(data)

load(data, './target.csv')