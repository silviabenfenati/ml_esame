import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st


def main():
    def main():

     st.title("LINEAR REGRESSION")

    x1rand = st.slider('Number of points: ', 1, 1000, 100)

    gen_random = np.random.RandomState(667)
    x = 10 * gen_random.rand(x1rand)

    coeff_ang = st.slider('Coefficente angolare: ', 0, 100, 1)
    noiserand = st.slider('Noise: ', 0, 100, 1)
    noise = np.random.normal(0,noiserand,x1rand)

    y = coeff_ang * x + noise

    X = x.reshape(-1, 1) # X--> features

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    fig = plt.figure(figsize = (10, 8))
    
    plt.scatter(x, y)           # in blu rappresento i punti
    plt.plot(x, y_pred,'-r')    # in rosso la retta di regressione 
    plt.title('Simple Linear Regression')
    plt.xlabel('Hours of study')
    plt.ylabel('Test scores')
    st.pyplot(fig)


if __name__ =="__main__":
    main()

