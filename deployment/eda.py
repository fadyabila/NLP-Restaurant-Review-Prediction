import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# Melebarkan visualisasi untuk memaksimalkan browser
st.set_page_config(
    page_title='Restaurant Review Prediction',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    # Membuat title
    st.title('Restaurant Review Prediction')
    st.write('### by Fadya Ulya Salsabila')

    # Menambahkan Gambar
    image = Image.open('review.png')
    st.image(image, caption='Customer Satisfaction')

    # Menambahkan Deskripsi
    st.write('## Background')
    st.write("""
    A restaurant bar in the Downtown Las Vegas area called Duck Vegas, has collected the results of reviews (reviews) from consumers or customers. The results of this review are a questionnaire from Customer Satisfaction or customer satisfaction.
    According to Kotler and Keller (2009), customer satisfaction is a person's feeling of pleasure or disappointment that arises from comparing the perceived performance of a product (or result) against their expectations.
    Duck Vegas will analyze customer satisfaction results to improve the performance and service of its restaurants.
    This is intended to create satisfaction that can provide several benefits including the relationship between the company and the customer to be harmonious, become the basis for repeat purchases, and create customer loyalty as well as word of mouth recommendations that benefit the company, in this case, Duck Vegas. The strategy and steps to create Customer Value and Customer Satisfaction are efforts to create loyal customers for this restaurant [1][2].
    Therefore, NLP (Natural Language Processing) analysis and modeling will be carried out using an Artificial Neural Network (ANN) to find out what types of words or text are included in the likes and dislikes reviews of restaurant services.
    So that Duck Vegas can implement the right strategy to increase its business.""")

    st.write('## Dataset')
    st.write("""
    The dataset is from [Kaggle](https://www.kaggle.com/datasets/vigneshwarsofficial/reviews?datasetId=277570&sortBy=voteCount) namely `Restaurant_Reviews.tsv`. 
    This dataset is reviews from 1000's of customers and contains `2 columns`, such as:

    1. `Review`: contains customer reviews review, in the form of text.
    2. `Liked`: rating of the review (0=Dislike and 1=Like).""")

    # Membuat Garis Lurus
    st.markdown('---')

    # Membuat Sub Headrer
    st.subheader('EDA for Restaurant Review')

    # Magic Syntax
    st.write(
    ' On this page, the author will do a simple exploration.'
    ' The dataset used is the Restaurant Review dataset.'
    ' This dataset comes from Kaggle.')

    # Show DataFrame
    df1 = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    st.dataframe(df1)

    # Membuat Barplot
    st.write('#### Liked Rating Plot')
    fig = plt.figure(figsize=(10,7))
    sns.countplot(x='Liked', data=df1, palette="PuRd")
    st.pyplot(fig)
    st.write('The target data is balanced.')

    # Menambahkan Gambar
    image = Image.open('wordcloud.png')
    st.image(image, caption='WordCloud Restaurant Review')

    # Membuat Histogram Berdasarkan Input User
    st.write('#### Histogram Based On User Input')
    pilihan = st.selectbox('Choose Column : ', ('Review', 'Liked'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(df1[pilihan], bins=30, kde=True)
    st.pyplot(fig)

if __name__ == '__main__':
    run()