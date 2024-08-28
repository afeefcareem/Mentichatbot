import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords') 

df = pd.read_csv('survey.csv')
#print(df.columns)

def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)
    else:
        return ''

text_columns = df.select_dtypes(include=['object']).columns

def preprocessed_columns(df, columns):
    for column in columns:
        df[f'cleaned_{column}'] = df[column].apply(preprocess_text)
    return df

df = preprocessed_columns(df, text_columns)

print(df[[f'cleaned_{col}' for col in text_columns]].head())

df.to_csv('preprocessed_survey_data.csv', index=False)

