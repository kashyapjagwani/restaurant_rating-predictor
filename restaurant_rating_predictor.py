import pandas as pd
from pandas._config.config import options
import streamlit as st
import numpy as np

header = st.container()
dataset = st.container()
modelTraining = st.container()
predictor = st.container()

# static content
df = pd.read_csv('Swiggy Bangalore Outlet Details.csv')
cuisine_options = []
for cuisines in df.Cuisine:
    for cuisine in cuisines.split(', '):
        cuisine_options.append(cuisine)
cuisine_options = np.unique(cuisine_options)

location_options = df.Location.unique()

# the cleaning function
def data_cleaning(df):
    # first, I will split location and extract only the locality name and discard the street address
    df['Locality'] = df.Location.str.rsplit(',',1,expand=True)[1]

    # next I will remove the '₹' symbol from Cost_for_Two
    df.Cost_for_Two = df.Cost_for_Two.map(lambda x: x.lstrip('₹ '))
    
    # now, I will split Cuisine into separate columns - basically like dummy encoding
    index = 0
    for cuisine in df.Cuisine:
        for c in cuisine.split(', '):
            df.loc[index, c] = 1
        index += 1
    
    # lastly, since Locality is also categorical, I will use one-hot encoding to separate it into encoded columns
    df = pd.get_dummies(df, columns = ['Locality'])
    
    # using fillna() to convert all NaN to 0
    df = df.fillna(0)

    # now the data is ready

    # remove the unwanted columns and break the data into dependant and independant variables

    # independant var
    X = df.drop(['Shop_Name', 'Cuisine', 'Location', 'Rating'], axis=1)

    # dependant var
    y = df.Rating

    return X, y


np.unique(cuisine_options)

with header:
    st.title('Welcome to Amazing Restaurant Rating Predictor')
    st.text('In this project I look into the features like location, cuisine served, price, \nand user ratings of various restaurants partnered with a food delivery \napp called Swiggy.')

with dataset:
    st.title('Swiggy dataset of restaurants')
    st.write(df.head())

with modelTraining:
    st.header('Creating the model')
    st.text('Choose the hyperparameters of the Random Forest and visualize its performance')

    sel_col, disp_col = st.columns(2)

    max_depth =sel_col.slider('Choose the max depth of the tree?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('Select the number of trees', options=[100, 200, 300])

    ## Dataset cleaning

    X,y = data_cleaning(df)

    ## Split the data 80-20 for training and testing the model

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # using pd.to_numeric() to convert all values in y_train to numeric

    y_train = pd.to_numeric(y_train, errors="coerce")
    y_train = y_train.fillna(0)

    ## Training the model

    # random forest
    from sklearn.ensemble import RandomForestRegressor
    randomForestModel = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    randomForestModel.fit(X_train, y_train)
    yPredForest = randomForestModel.predict(X_test)

    ## Evaluating the model

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y_test, yPredForest))
    
    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y_test, yPredForest))

    disp_col.subheader('RMSE of the model is:')
    disp_col.write(mean_squared_error(y_test, yPredForest, squared=False))

with predictor:
    st.header('Enter your restaurant\'s features, and the model will predict you user rating out of 5')

    location = st.selectbox('Location of your restaurant', options=location_options)
    cuisines = st.multiselect('Cuisines you offer', options=cuisine_options)
    cost = st.number_input('Cost for two')
    predict = st.button('Predict Rating')

    if(predict):
        df = df.append({
            'Cuisine': ', '.join(cuisines),
            'Location': location,
            'Cost_for_Two': '₹ ' + str(int(cost))
        }, ignore_index=True)
        new_value_X, new_value_y= data_cleaning(df)
        predicted_rating = randomForestModel.predict(new_value_X.tail(1))[0]
        st.subheader('Your restaurant will receive a rating of around ' + str(round(predicted_rating, 1)))
    
