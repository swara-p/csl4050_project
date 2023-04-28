import streamlit as st
import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score, roc_curve, f1_score, classification_report, confusion_matrix
from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Algorithm Comparison App',
    layout='wide')
#---------------------------------#
# Model building

def build_model(df):
    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y


    st.markdown('**1.2. Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable (first 20 are shown)')
    st.info(list(X.columns[:20]))
    st.write('Y variable')
    st.info(Y.name)

    # Build lazy model
    if task=="Classification":
        le=LabelEncoder()
        le.fit(Y)
        Y=le.transform(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = split_size)

    if task=="Regression":
        reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
        models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
        models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

        bins = st.slider('Number of bins:', 1, 100, 50, 1)
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax5 = sns.histplot(x=df.columns[-1],data=df,bins=bins,color="#69b3a2")
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-targetDistribution.pdf'), unsafe_allow_html=True)


    elif task=="Classification":
        clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        models_train,predictions_train = clf.fit(X_train, X_train,  Y_train, Y_train)
        models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)

    st.subheader('2. Table of Model Performance')

    st.write('Training set')
    st.write(predictions_train)
    st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

    st.write('Test set')
    st.write(predictions_test)
    st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

    st.subheader('3. Plot of Model Performance (Test set)')
    if task=="Regression":

        with st.markdown('**R-squared**'):
            # Tall
            predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]

        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-r2.pdf'), unsafe_allow_html=True)

        with st.markdown('**RMSE (capped at 200)**'):
            # Tall
            predictions_test["RMSE"] = [200 if i > 200 else i for i in predictions_test["RMSE"] ]
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-rmse.pdf'), unsafe_allow_html=True)

        with st.markdown('**Calculation time**'):
            # Tall
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]

        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-calculation-time.pdf'), unsafe_allow_html=True)

    if task=="Classification":
        with st.markdown('**Accuracy**'):
            # Tall
            predictions_test["Accuracy"] = [0 if i < 0 else i for i in predictions_test["Accuracy"] ]
    #         # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="Accuracy", data=predictions_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-acc.pdf'), unsafe_allow_html=True)

        with st.markdown('**Balanced Accuracy**'):
            # Tall
            predictions_test["Balanced Accuracy"] = [0 if i < 0 else i for i in predictions_test["Balanced Accuracy"] ]
    #         # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="Balanced Accuracy", data=predictions_test)
        ax2.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-bal_acc.pdf'), unsafe_allow_html=True)

        with st.markdown('**F1 Score**'):
            # Tall
            predictions_test["F1 Score"] = [0 if i < 0 else i for i in predictions_test["F1 Score"] ]
    #         # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="F1 Score", data=predictions_test)
        ax3.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-fsore.pdf'), unsafe_allow_html=True)

        with st.markdown('**Calculation time**'):
            # Tall
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
            # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax4 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-calculation-time.pdf'), unsafe_allow_html=True)


# Download CSV data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#
st.write("""
# ML Algorithm Comparison Dashboard 
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe

with st.sidebar.header('1. Choose your Task'):
    task = st.sidebar.radio('Task', ('Regression', 'Classification'), horizontal=True, label_visibility='collapsed')


with st.sidebar.header('2. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")


with st.sidebar.header('3. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

#---------------------------------#
# Main panel

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button(f'Press to use Example Dataset for {task}'):
        if task == 'Classification':
          df = load_iris()
          print()
          name = 'Iris'
        else:
          df = load_diabetes()
          name = 'Diabetes'
        st.markdown(f'The {name} dataset is used as the {task} task example.')
        df = sklearn_to_df(df)
        st.write(df.head(5))
        build_model(df)

