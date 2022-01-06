import pandas as pd
import streamlit as st

header = st.container()
dataset = st.container()
modelTraining = st.container()

with header:
    st.title('Welcome to Amazing Restaurant Rating Predictor')
    st.text('In this project I look into the features like location, cuisine served, price, \nand user ratings of various restaurants partnered with a food delivery \napp called Swiggy.')

with dataset:
    st.title('Swiggy dataset of restaurants')
    df = pd.read_csv('Swiggy Bangalore Outlet Details.csv')
    st.write(df.head())

with modelTraining:
    st.header('Creating the model')
    st.text('Choose the hyperparameters of the Random Forest and visualize its performance')

    sel_col, disp_col = st.columns(2)

    max_depth =sel_col.slider('Choose the max depth of the tree?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('Select the number of trees', options=[100, 200, 300])

    ## Dataset cleaning

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

    ## Split the data 80-20 for training and testing the model

    from sklearn.model_selection import train_test_split

    X = df.drop(['Shop_Name', 'Cuisine', 'Location', 'Rating'], axis=1)
    y = df.Rating

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