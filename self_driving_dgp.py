import numpy as np
import scipy as sp
import pandas as pd

from numpy.random import normal as rnd
from scipy.special import expit
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
from dataclasses import dataclass


class DGP_SELFDRIVING():
    """
    Data generating process: self-driving cars
    """

    def clean_data():
        df = pd.read_csv('/home/ruifan/environments/synthetic_control/us_cities_20022019.csv')
        df = df[['Metropolitan areas', 'Variables', 'Year', 'Value']]
        df.columns = ['city', 'variable', 'year', 'value']
        df.loc[df['variable'].str.contains('Employment'), "variable"] = "employment"
        df.loc[df['variable'].str.contains('Population density'), "variable"] = "density"
        df.loc[df['variable'].str.contains('Population'), "variable"] = "population"
        df.loc[df['variable'].str.contains('GDP'), "variable"] = "gdp"
        df = pd.pivot(data=df, index=['city', 'year'], columns='variable').reset_index()
        df.columns = [''.join(col).replace('value', '') for col in df.columns]
        df['employment'] = df['employment'] / df['population']
        df['city'] = df['city'].str.replace('\(\w+\)', '').str.strip()
        df['population'] = df['population'] / 1e6
        df['gdp'] = df['gdp'] / 1e4
        df.to_csv('/home/ruifan/environments/synthetic_control/us_cities_20022019_clean.csv', index=False)


    def generate_data(self, city='Chicago', year=2010, seed=1):
        np.random.seed(seed)

        # Load Data
        df = pd.read_csv('/home/ruifan/environments/synthetic_control/us_cities_20022019_clean.csv')
        df = df[df['year']>2002]

        # Select only big cities
        df['mean_pop'] = df.groupby('city')['population'].transform('mean')
        df = df[df['mean_pop'] > 1].reset_index(drop=True)
        del df['mean_pop']

        # Treatment
        df['treated'] = df['city']==city
        df['post'] = df['year']>=year

        # Generate revenue
        df['revenue'] = df['gdp'] + np.sqrt(df['population']) + \
            20*np.sqrt(df['employment']) - df['density']/100 + \
            (df['year']-1990)/5 + np.random.normal(0,1,len(df)) + \
            df['treated'] * df['post'] * np.log(np.maximum(2, df['year']-year))

        return df

if __name__ == "__main__":
    DGP_SELFDRIVING.clean_data()