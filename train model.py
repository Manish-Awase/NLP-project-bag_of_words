import pandas as pd
import spacy
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from joblib import dump

model_path='./artifact/model/trained_model.joblib'
vectorizer_path = './artifact/vectorizer/vectorizer.joblib'
# os.makedirs(os.path.dirname(model_path), exist_ok=True)
# os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

def preprocessing_step(sentence):
    sentence = sentence.lower()
    spacy_nlp = spacy.load('en_core_web_sm')
    doc = spacy_nlp(sentence)
    sentence=[token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(sentence)


spam_labels = {
    "spam": 1,
    "ham": 0,

}
# data collection
df = pd.read_csv('resources/spam.csv')

# preprocessing
df["spam"]=df["Category"].map(spam_labels)

df["Message_preprocessed"]=df["Message"].apply(preprocessing_step)

#split
X_train, X_test ,y_train, y_test= train_test_split(df["Message_preprocessed"].values, df["spam"].values,test_size=0.2, random_state=42)

# vectorization
vectorizer = CountVectorizer()

X_train_v= vectorizer.fit_transform(X_train)
X_test_v= vectorizer.transform(X_test)

# training

model =MultinomialNB()
model.fit(X_train_v,y_train)

y_predict = model.predict(X_test_v)
print(classification_report(y_test,y_predict))

# save model,vectorizer

# 'model' is your trained ML model
dump(model, model_path)
dump(vectorizer, open(vectorizer_path, 'wb'))
print('model Saved')
