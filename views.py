
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.feature_extraction.text import CountVectorizer


import json,nltk
import string
import pandas as pd
import numpy as np

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import random
nltk.download()

import warnings
warnings.filterwarnings('ignore')
import json 
with open('data_base.json') as json_file:
    data1 = json.load(json_file)
    
data = data1['intents']

df = pd.DataFrame(columns=['intent', 'questions', 'answers'])
for i in data:
    intent = i['intent']
    for t, r in zip(i['questions'], i['answers']):
        row = {'intent': intent, 'questions': t, 'answers':r}
        df = df.append(row, ignore_index=True)

 #Checking missing value in label and text
print('\nTotal missing values:\n', df.iloc[:, 0:-1].isnull().sum(), "\n")


df['questions'] = df['questions'].astype(str)
#df['text_clean'] = df['text_clean'].astype(str)


df['question'] = df['questions'].str.lower()


import contractions

df["question"] = df["question"].apply(contractions.fix)

from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
", ".join(stopwords.words('english'))

stopwords_to_remove = set(stopwords.words('english'))


def remove_stopwords(text: str, _stopwords: set):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in _stopwords])


df["question"] = df["question"].apply(lambda text: remove_stopwords(text, stopwords_to_remove))


import string

punctuation_to_remove = string.punctuation
print('We\'ll remove', punctuation_to_remove)


def remove_punctuation(text: str, punctuation: str):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', punctuation))



df['question'] = df['question'].apply(lambda text: remove_punctuation(text, punctuation_to_remove))

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

lemmatizer_to_use = WordNetLemmatizer()
words_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}


def lemmatize_words(text, lemmatizer: WordNetLemmatizer, wordnet_map):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


df["questions"] = df["questions"].apply(lambda text: lemmatize_words(text, lemmatizer_to_use, words_map))

# Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

cv = CountVectorizer()
X = cv.fit_transform(df.question)

# Get the categories
y = df.intent

# Split arrays or matrices into 80%-20% train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(X_train.shape, " = x train shape")
print(X_test.shape, " = x test shape")


from sklearn.metrics import precision_score, accuracy_score
from time import perf_counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

 


    
def model_training():
    from joblib import dump, load

    try:
        
        model = load('ml_model.joblib')
        
    except:
            models = {
                "Random Forest": {"model": RandomForestClassifier(), "perf": 0},
                "Gradient Boosting": {"model": GradientBoostingClassifier(), "perf": 0},
                "MultinomialNB": {"model": MultinomialNB(), "perf": 0},
                "Logistic Regr.": {"model": LogisticRegression(), "perf": 0},
                "KNN": {"model": KNeighborsClassifier(), "perf": 0},
                "Decision Tree": {"model": DecisionTreeClassifier(criterion='gini', random_state=42), "perf": 0},
                "SVM (Linear)": {"model": LinearSVC(), "perf": 0},
                "SVM (RBF)": {"model": SVC(), "perf": 0}
            }

            for name, model in models.items():
                start = perf_counter()
                model['model'].fit(X_train, y_train)
                duration = perf_counter() - start
                duration = round(duration, 2)
                model["perf"] = duration
                y_pred = model['model'].predict(X_test)
                print(
                    f'{name:20} trained in {duration} sec, precision: {round(precision_score(y_test, y_pred, average="micro"), 3) * 100}%,'
                    f' accuracy: {round(accuracy_score(y_test, y_pred), 3) * 100}%')

            from sklearn.model_selection import cross_val_score
            from time import perf_counter
            from sklearn.svm import LinearSVC
            from sklearn.metrics import precision_score

            model = LinearSVC()

            start = perf_counter()
            model.fit(X_train, y_train)
            duration = round(perf_counter() - start, 3)

            # Input from test set
            y_pred = model.predict(X_test)

            print(
                f'## Precision average micro: {round(precision_score(y_test, y_pred, average="micro"), 3) * 100}%,\n'
                f'## Precision average macro: {round(precision_score(y_test, y_pred, average="macro"), 3) * 100}%,\n'
                f'## Accuracy: accuracy: {round(accuracy_score(y_test, y_pred), 3) * 100}, ({duration} secs)\n')

            print("Cross validation (precision)")
            print(cross_val_score(model, X, y, cv=5, scoring="precision_micro"))
                


        
            dump(model, 'ml_model.joblib')

            model = load('ml_model.joblib')
    
    return model

model = model_training()


from rest_framework.response import Response
from rest_framework.views import APIView

class Chatbot(APIView):

    '''
    *UPTodo* chatbot, POST requests with JSON data in the following format:
    {
        "query":"how to create a task"
    } 
    '''
    def post(self, request, format=None):
        data = request.data

        for key in data:
          message = data[key]
        
        print(message)

        predicted_label = model.predict(cv.transform([message]))[0]
        print(predicted_label)
        for tg in data1["intents"]:
                  
                if tg['intent'] == predicted_label:
                        answers = tg['answers']
                      
        
        response_dict= {"answer": random.choice(answers)}
        
        return Response(response_dict, status=200)













