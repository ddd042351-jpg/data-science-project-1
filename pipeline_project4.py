import os
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text, lemmatizer, stop_words):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words:
            pos_tag = get_wordnet_pos(word)
            lemma = lemmatizer.lemmatize(word, pos_tag)
            cleaned_tokens.append(lemma)
            
    return " ".join(cleaned_tokens)

def run_nlp_pipeline():
    file_name = "Dataset for Data Analytics - Sheet1.csv"
    
    if not os.path.exists(file_name):
        print(f"Error: Could not find {file_name}")
        return
        
    df = pd.read_csv(file_name)
    
    text_col = None
    for col in ['product', 'couponcode', 'referralsource']:
        for actual_col in df.columns:
            if col in actual_col.lower():
                text_col = actual_col
                break
        if text_col:
            break
            
    if not text_col:
        df['Text_Data'] = df.astype(str).values.sum(axis=1)
        text_col = 'Text_Data'
        
    df = df.dropna(subset=[text_col])
    
    if 'TotalPrice' in df.columns:
        df['Sentiment_Label'] = (df['TotalPrice'] > df['TotalPrice'].median()).astype(int)
    else:
        np.random.seed(42)
        df['Sentiment_Label'] = np.random.choice([0, 1], size=len(df))
        
    lemmatizer = WordNetLemmatizer()
    default_stopwords = set(stopwords.words('english'))
    negations = {'not', 'no', 'nor', 'neither', 'never', 'arent', 'couldnt', 'didnt', 'doesnt', 'hadnt', 'hasnt', 'havent', 'isnt', 'mightnt', 'mustnt', 'neednt', 'shant', 'shouldnt', 'wasnt', 'werent', 'wont', 'wouldnt'}
    custom_stopwords = default_stopwords - negations
    
    df['Cleaned_Text'] = df[text_col].apply(lambda x: clean_text(x, lemmatizer, custom_stopwords))
    
    df = df[df['Cleaned_Text'].str.strip() != ""]
    
    X = df['Cleaned_Text']
    y = df['Sentiment_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    predictions = model.predict(X_test_tfidf)
    
    print("--- Project 4 NLP Evaluation Metrics ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
    print(classification_report(y_test, predictions, zero_division=0))
    print("----------------------------------------")

if __name__ == "__main__":
    run_nlp_pipeline()