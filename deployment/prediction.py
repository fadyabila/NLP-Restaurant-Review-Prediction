import streamlit as st
import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import LancasterStemmer
from keras.models import load_model
import string

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('data')

# Load the pre-trained model
model = tf.keras.models.load_model('restaurant_review')

# Mendefinisikan Stopwords
stopwords_eng = stopwords.words("english")

# Membuat Stopwords Baru
new_stw = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"
 ]

# Compile Stopwords
stw_en = stopwords_eng + new_stw
stw_en = list(set(stw_en))

# Define Lancaster Stemmer
lanc = LancasterStemmer()

# Define the text processing function
def text_proses(review):
    # Mengubah Review ke Lowercase
    review = review.lower()
  
    # Menghilangkan Hashtag
    review = re.sub("#[A-Za-z0-9_]+", " ", review)
  
    # Menghilangkan \n
    review = re.sub(r"\\n", " ",review)
  
    # Menghilangkan Whitespace
    review = review.strip()

    # Menghilangkan Tanda Baca
    review = review.translate(str.maketrans('', '', string.punctuation))

    # Menghilangkan Link
    review = re.sub(r"http\S+", " ", review)
    review = re.sub(r"www.\S+", " ", review)

    # Menghilangkan yang Bukan Huruf seperti Emoji, Simbol Matematika (seperti μ), dst
    review = re.sub("[^A-Za-z\s']", " ", review)

    # Menghilangkan duplicate characters
    review = re.sub("(.)\\1{2,}", "\\1", review)

    # Merapikan Spasi
    review = ' '.join(review.split())

    # Melakukan Tokenisasi
    tokens = word_tokenize(review)

    # Menghilangkan Stopwords
    review = ' '.join([word for word in tokens if word not in stw_en])
  
    # Melakukan Stemming
    review = lanc.stem(review)
    
    # Join the processed words into a single string
    return review

# Define the lemmatization function
def lemmatize_text(text):
    sentence = []
    for word in text.split():
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))
    return ' '.join(sentence)

# Define the Streamlit app
def run():
    st.title("Restaurant Review Inference")
    
    # Define the input form
    review = st.text_input("Enter a restaurant review:")
    liked = st.selectbox("Did you like the restaurant?", ['Yes', 'No'])
    
    # Process the input data
    data = {
        'Review': review,
        'Liked': 1 if liked == 'Yes' else 0
    }
    data = pd.DataFrame([data])
    data['text_processed'] = data['Review'].apply(text_proses)
    data['text_processed'] = data['text_processed'].apply(lemmatize_text)
    
    # Perform the inference
    if st.button("Predict"):
        y_pred = model.predict(data['text_processed'])
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        
        # Show the prediction result
        if y_pred[0] == 1:
            st.write("The review is positive!")
        else:
            st.write("The review is negative.")

# Run the Streamlit app
if __name__ == '__main__':
    app()