import streamlit as st
import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_diabetes, load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import streamlit.components.v1 as components
import pandas_profiling

st.set_page_config(page_title='DashML', layout='wide')

def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

def build_model(df):
    X = df.iloc[:,:-1] 
    if target_col != '':
        Y = df[target_col]
    else:
        Y = df.iloc[:,-1] 

    st.markdown('**1.2. Dataset dimension**')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = split_size/100)

    cols = st.columns(8, gap='small')
    cols[0].markdown('***X_train***')
    cols[1].write(X_train.shape)
    cols[2].markdown('***X_test***')
    cols[3].write(X_test.shape)
    cols[4].markdown('***Y_train***')
    cols[5].write(Y_train.shape)
    cols[6].markdown('***Y_test***')
    cols[7].write(Y_test.shape)

    st.markdown('**1.3. Variable details**:')
    cols = st.columns([1,5], gap='small')
    cols[0].markdown('***Features (X)***')
    cols[1].info(list(X.columns))

    cols = st.columns([1,5], gap='small')
    cols[0].markdown('***Target (Y)***')
    cols[1].info(Y.name)

    if task=="Classification":
        le=LabelEncoder()
        le.fit(Y)
        Y=le.transform(Y)

    st.subheader('2. EDA Report')
    with st.spinner('Wait for it...'):
        profile = pandas_profiling.ProfileReport(df)
        profile.to_file("report.html")
        HtmlFile = open("report.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height=5*600)

    st.subheader('3. Model Performance')
    with st.spinner('Wait for it...'):
        if task=="Regression":
            reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
            _, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
            _, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)
 
        elif task=="Classification":
            clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
            _, predictions_train = clf.fit(X_train, X_train,  Y_train, Y_train)
            _, predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)

        st.write('Training set')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train,'training.csv'), unsafe_allow_html=True)

        st.write('Test set')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test,'test.csv'), unsafe_allow_html=True)

    st.subheader('4. Metrics Plots (on Test set)')
    if task=="Regression":

        with st.markdown('**R-squared**'):
            predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"] ]
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-r2.pdf'), unsafe_allow_html=True)

        with st.markdown('**RMSE (capped at 200)**'):
            predictions_test["RMSE"] = [200 if i > 200 else i for i in predictions_test["RMSE"] ]
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-rmse.pdf'), unsafe_allow_html=True)

        with st.markdown('**Calculation time**'):
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-calculation-time.pdf'), unsafe_allow_html=True)

    if task=="Classification":

        with st.markdown('**Accuracy**'):
            predictions_test["Accuracy"] = [0 if i < 0 else i for i in predictions_test["Accuracy"] ]
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="Accuracy", data=predictions_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-acc.pdf'), unsafe_allow_html=True)

        with st.markdown('**Balanced Accuracy**'):
            predictions_test["Balanced Accuracy"] = [0 if i < 0 else i for i in predictions_test["Balanced Accuracy"] ]
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="Balanced Accuracy", data=predictions_test)
        ax2.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-bal_acc.pdf'), unsafe_allow_html=True)

        with st.markdown('**F1 Score**'):
            predictions_test["F1 Score"] = [0 if i < 0 else i for i in predictions_test["F1 Score"] ]
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="F1 Score", data=predictions_test)
        ax3.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-fsore.pdf'), unsafe_allow_html=True)

        with st.markdown('**Calculation time**'):
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"] ]
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax4 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt,'plot-calculation-time.pdf'), unsafe_allow_html=True)

st.write("""# DashML: ML Algorithm Comparison Dashboard """)

with st.sidebar.header('1. Choose your Task'):
    task = st.sidebar.radio('Task', ('Regression', 'Classification'), horizontal=True, label_visibility='collapsed')

with st.sidebar.header('2. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"], label_visibility='collapsed')
    target_col = st.sidebar.text_input('Enter the target column name if it is not the last column')
    normalize = st.sidebar.checkbox('Check to normalize the continuous features in the data', False)

with st.sidebar.header('3. Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

st.subheader('1. Dataset')

if 'eg' not in st.session_state:
    st.session_state.eg = False
    
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df.head(5))
    build_model(df)
else:
    placeholder = st.empty()
    with placeholder.container():
        egg = st.button(f'Press to use Example Dataset for {task}')
    if egg or st.session_state.eg:
        st.session_state.eg = True
        placeholder.empty()
        if task == 'Classification':
          df = load_iris()
          name = 'Iris'
        else:
          df = load_diabetes()
          name = 'Diabetes'
        st.markdown(f'The {name} dataset is used as the {task} task example.')
        df = sklearn_to_df(df)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df.head(5))
        build_model(df)
