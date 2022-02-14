import pandas as pd
import streamlit as st 
import numpy as np
from PIL import Image
import plotly.express as px

image = Image.open('D:\\Programming\\Code Workplace\\Python\\MyProjects\\Machine-Learning-Algorithms-Thuật-toán-Machine-Learning.png')
img = st.image(image, use_column_width=True)

st.sidebar.title("Choose Machine Learning Model")
model_select = ['Home','Simple Linear Regression','Multivariable Linear Regression',
                'Polynomial Regression','Decision Tree','Logistic Regression','SVM','Naive Bayes']

choice = st.sidebar.selectbox('Select model...', model_select)


if choice == 'Simple Linear Regression':
    img.empty()

    st.title("Welcome to Simple Linear Regression Application")
    st.subheader("Description")
    st.write("""
    The data is from `Advertising.csv`, we combine 3 different platforms into **total_spending** 
    to predict total sales.
    """)
    st.write("##### Here is our dataset")

    from Simple_Linear_Regression import df, X , y, X_test, y_test  
    from Simple_Linear_Regression import fig, fig_train, fig_test, fig_fit
    from Simple_Linear_Regression import rmse, mean_ab_er, final_model, rsquare
    st.write(df)
    
    # Scatter Plot
    if st.checkbox("Scatter Plot"):
        full_plot = st.plotly_chart(fig)

        choose_plot = st.selectbox('Choose one:',['Default','Train','Test'])
        if choose_plot == 'Train':
            st.plotly_chart(fig_train)

        if choose_plot == 'Test':
            st.plotly_chart(fig_test)

    # Test set
    if st.checkbox('Test Set'):
        data = {'total_spend': X_test,'sales': y_test}
        table = pd.DataFrame(data)
        st.write(table)
    # Predict Sales
    if st.checkbox('Predict and Evaluate'):
       
        st.plotly_chart(fig_fit)

        number = st.number_input('Insert total spending:',step= 50.0)
        st.write('You choose: ', number)

        input_number = np.array(number).reshape(1,-1)
        pred = final_model.predict(input_number)
        st.write('#### Predicted Sales:' , pred)

        st.write('**Mean Absolute Error:** ', mean_ab_er)
        st.write('**Root Mean Square Error:**', rmse)
        st.write('**R-Squared value:**', rsquare)
    
elif choice == 'Multivariable Linear Regression':
    img.empty()
    st.title("Welcome to Multivariable Linear Regression Application")
    st.subheader("Description")
    st.write("""
    The data is from `Advertising.csv`, we will consider the money spend on 3 different advertising
    platforms to predict sales.
    """)
    st.write("##### Here is our dataset")
    
    from Multi_Linear_Regression import df, X_test, y_test, df_test_set
    from Multi_Linear_Regression import fig, residual_plot, TV_hist, radio_hist, news_hist, sales_hist
    from Multi_Linear_Regression import final_model, mae, rmse, rsquare
    st.write(df)

    # Correlation Plot
    if st.checkbox('Correlation Plot'):
        st.pyplot(fig)
    
    # Histogram
    if st.checkbox('Histogram'):
        st.write('''A histogram shows the number of instances (on the vertical axis) that have a 
        given value range (on the horizontal axis).
        ''')
        st.plotly_chart(TV_hist)
        st.plotly_chart(news_hist)
        st.plotly_chart(radio_hist)
        st.plotly_chart(sales_hist)

    # Residual Plot
    if st.checkbox('Residual Plot'):
        st.write('''
        A residual plot is a type of scatter plot that shows the residuals on the vertical axis 
        and the independent variable on the horizontal axis. Explore the definition and examples 
        of residual plots and learn about the sum of squared residuals
        ''')
        st.plotly_chart(residual_plot)

    if st.checkbox('Test Set'):
        st.write(df_test_set)

    if st.checkbox('Predict and Evaluate'):

        # Get user input function
        def user_input_features():
           
            TV_spending = st.sidebar.slider('TV Spending',0,700,300)
            radio_spending = st.sidebar.slider('Radio Spending',0,100,50)
            news_spending = st.sidebar.slider('Newspaper Spending',0,100,50)
            
            data = {'TV Spending': TV_spending,
                    'Radio Spending':radio_spending,
                    'Newspaper Spending': news_spending}

            features = pd.DataFrame(data, index=[0])
            return features

        # User input
        user_input = user_input_features()

        # Predict
        pred_sales = final_model.predict(user_input)
        st.write('#### Predicted Sales', pred_sales)

        # Evaluation
        st.write('**Mean Absolute Error:** ', mae)
        st.write('**Root Mean Square Error:**', rmse)
        st.write('**R-Squared value:**', rsquare)
    

elif choice == 'Decision Tree':
    img.empty()
    st.title("Welcome to Decision Tree Application")
    st.subheader("Description")
    st.write("""
    The data is from `penguins_size.csv`, we will predict the species of given penguins (Adelie, Chinstrap
    , Gentoo) base on their features.
    """)
    st.write("##### Here is our dataset")   

    from Decision_Tree import df, number_of_species, number_of_null

    st.write(df)
    st.write('''We can see that the data is not clean. There are **NA** values and some **noises**.
    Let's explore the data !
    ''')

    if st.checkbox('Explore'):
        st.markdown('''
        First let's see how many species that we need to classify
        ''')
        st.code("df['species'].unique()")
        st.write(number_of_species)

        st.write("Next let's see how many **NA** value that we have")
        st.code("df.isnull().sum()")
        st.write('''Here we take the sum because NA value return 1, non-NA return 0, so taking
        the sum will gives us how many NA values
        ''')
        st.write(number_of_null)
        st.write("""Here we see that at following features **culmen_length_mm**, **culmen_depth_mm**, 
        **flipper_length_mm**, **body_mass_g** we have 2 NA values. 
        """)

