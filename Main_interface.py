import pandas as pd
import streamlit as st 
import numpy as np
from PIL import Image
import plotly.express as px

image = Image.open('D:\\Programming\\Code Workplace\\Python\\MyProjects\\ML_Web_App\\image\\ml.png')
img = st.image(image, use_column_width=True)

st.sidebar.title("Choose Machine Learning Model")
model_select = ['Home','Simple Linear Regression','Multivariable Linear Regression',
                'Polynomial Regression','Decision Tree','Logistic Regression','SVM','Naive Bayes',
                'Random Forest']

choice = st.sidebar.selectbox('Select model...', model_select)

###################################################################

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


###################################################################

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

###################################################################
elif choice == 'Polynomial Regression':
    img.empty()
    st.info('Oops! Polynomial Regression is not available yet. Please come back later!')






###################################################################

elif choice == 'Decision Tree':
    img.empty()
    st.title("Welcome to Decision Tree Application")
    st.subheader("Description")
    st.write("""
    The data is from `penguins_size.csv`, we will predict the species of given penguins (Adelie, Chinstrap
    , Gentoo) base on their features.
    """)
    st.write("##### Here is our dataset")   

    from Decision_Tree import df, number_of_species, number_of_null, df_new
    from Decision_Tree import island_cate, sex_cate, dot_in_sex, sex_and_others_relation, check_sex
    
    if st.checkbox("Original dataset"):
        st.write(df)

        st.write('''We can see that the data is not clean. There are **NA** values and some **noises**.
        Let's explore the data !
        ''')

    if st.checkbox('Explore and Preprocessing data'):
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

        st.write("Let's see the dataset after removing **NA** values")
        st.code('df_new = df.dropna()')
        st.write(df_new)

        st.write("Let's check if there are some noises remaining")
        st.code("df_new['island'].unique()")
        st.code("df_new['sex'].unique()")
        st.write(island_cate)
        st.write(sex_cate)

        st.write("Oops! We have a noisy value in 'sex' columns. Let's check where it is")
        st.code("df_new[df_new['sex'] == '.']")
        st.write(dot_in_sex)

        st.write("It seems that we need to replace this 'dot' with another gender")
        st.write("""But first, let's see the relationship between other features and sex (just consider 
        Gentoo species)
        """)
        st.code("df_new[df_new['species'] == 'Gentoo'].groupby('sex').describe().transpose()")
        st.write(sex_and_others_relation)

        st.write("""In this case, it seems to be female rather than male (just personally). So
        let's replace it with **'FEMALE'**""")
        st.code("df_new.at[336,'sex'] = 'FEMALE'") 
        st.write("Let's check if it has changed")
        st.write(check_sex.astype(str))

        st.write("""HOORAY! It has been changed. Let's take a look at our after-processing dataset
        """)

    if st.checkbox("After-processing data"):
        st.dataframe(df_new)
    
    from Decision_Tree import pair_fig, cat_fig
    from Decision_Tree import X_test, y_test, df_test_set,X_train,y_train
    from Decision_Tree import report_model

    
    if st.checkbox("Pair Plot"):
        st.pyplot(pair_fig)

    if st.checkbox("Categorical Plot"):
        st.pyplot(cat_fig)

    if st.checkbox("Test Set"):
        st.dataframe(df_test_set)

    if st.checkbox("Prediction and Evaluation"):
        st.sidebar.subheader('Features Selection')
        
        with st.sidebar:
            with st.form('Form1'):
                island = st.selectbox('Island',
                                    ['Torgersen','Biscoe','Dream'])
                culmen_length = st.number_input('Culmen_length_mm',0.0,100.0, step= 1.0)
                culmen_depth = st.number_input('Culmen_depth_mm',0.0,50.0, step= 1.0)
                flipper_length = st.number_input('Flipper_length_mm',0.0,250.0, step= 1.0)
                body_mass = st.number_input('Body_mass_g',0.0,7000.0, step= 1.0)     
                sex = st.selectbox('Sex', ['FEMALE','MALE'])     

                submitted = st.form_submit_button("Predict")

        data = {'island': island,
                'culmen_length_mm': culmen_length,
                'culmen_depth_mm': culmen_depth,
                'flipper_length_mm': flipper_length,
                'body_mass_g': body_mass,
                'sex': sex}

        features = pd.DataFrame(data, index=[0])
        
        # Encode user_input
        penguins = df_new.drop('species', axis=1)  # Use the cleaned dataframe, not the get_dummies one
        user_input = pd.concat([features,penguins], axis=0)
    
        encode = ['island','sex']
        for col in encode:
            dummy = pd.get_dummies(user_input[col], prefix= col, drop_first= True)
            user_input = pd.concat([user_input,dummy], axis=1)
            del user_input[col]
        user_input = user_input[:1]
        
        st.subheader('Your choice')
        st.write(features)
        st.write('After get_dummies')
        st.write(user_input)

        # Predict
        from Decision_Tree import tuning_model, train_model, show_metrics, cfm_plot

        st.subheader('Tuning Hyperparameter')
        model = tuning_model()
        model = train_model(model,X_train,y_train) 
        y_pred, fea_importance = show_metrics(model,X_test)

        if submitted:
            prediction = model.predict(user_input)
            prediction_proba = model.predict_proba(user_input)

            st.write("#### Predicted: ")
            st.write(prediction)
            st.write("#### Prediction Probability")
            st.write(prediction_proba)

        if st.checkbox("Features Importances"):
            st.dataframe(fea_importance)

        if st.checkbox("Confusion Matrix"):
            confu_matrix = cfm_plot(y_test,y_pred)
            st.pyplot(confu_matrix)

        if st.checkbox("Report model and plot tree"):
            report_model(model)


###################################################################
elif choice == 'Logistic Regression':
    img.empty()
    st.info('Oops! Logistic Regression is not available yet. Please come back later!')



###################################################################
elif choice == 'SVM':
    img.empty()
    st.info('Oops! SVM is not available yet. Please come back later!')



###################################################################
elif choice == 'Naive Bayes':
    img.empty()
    st.info('Oops! Naive Bayes is not available yet. Please come back later!')




###################################################################
elif choice == 'Random Forest':
    img.empty()
    st.info('Oops! Random Forest is not available yet. Please come back later!')
