###Source data/upload data
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

Notes
print(type(text_data))
print(text_data.keys())
print(text_data.target_names)
a buch can't be analysis
import numpy as np
from sklearn.datasets import fetch_20newsgroups

text_data = fetch_20newsgroups()
raw_text = text_data.data[:4]
raw_text
###Data Cleaning start with - stage 1- convert text to lowercase
clean_text_1 = []
def to_lower_case(data):
  for words in raw_text:
    clean_text_1.append(str.lower(words))
to_lower_case(raw_text)
clean_text_1
###stage 2- tokenization
clean_text_2 = []
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab') # Added this line to download punkt_tab
for text_item in clean_text_1:
  sent = sent_tokenize(text_item)
  clean_text_2.append(sent)

print(clean_text_2[0])
clean_text_3 = [] # Initialize the list
for sentence_list in clean_text_2:
  word_list = []
  for sentence in sentence_list:
    words = word_tokenize(sentence)
    word_list.extend(words)
  clean_text_3.append(word_list)

print(clean_text_3[0])
clean_text_2 = [sent_tokenize(i) for i in clean_text_1]
clean_text_2
import re
clean_text_4 = []
for words in clean_text_2:
  clean = []
  for w in words:
    res = re.sub(r'[^\w\s]','',w)
    if res != '':
      clean.append(res)
    clean_text_4.append(clean)




###stage 3- special charcter removal
clean_text_4
###stage 4-stopword removal

> Add blockquote


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

clean_text_5 = []
stop_words = stopwords.words('english')

for words in clean_text_4:
  w = []
  for word in words:
    if word not in stop_words:
      w.append(word)
  clean_text_5.append(w)
clean_text_5
###stage 5-stemming
from nltk.stem import PorterStemmer
port = PorterStemmer()
clean_text_6 = []
for words in clean_text_5:
  w = []
  for word in words:
    w.append(port.stem(word))
  clean_text_6.append(w)
clean_text_6
###Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
wnet = WordNetLemmatizer()
lem = []
for words in clean_text_5:
  w = []
  for word in words:
    w.append(wnet.lemmatize(word))
  lem.append(w)

lem
print(lem)

import pandas as pd

df = pd.DataFrame({'processed_text': lem})
display(df.head())
df.shape

df.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import pandas as pd

# Ensure df is a DataFrame. If not, reconstruct it or raise an error.
# Assuming 'lem' is the source of the text data, we can reconstruct df if needed
if not isinstance(df, pd.DataFrame):
    print("Warning: df is not a DataFrame. Reconstructing from 'lem'.")
    # Assuming 'lem' has one element per original text entry
    # We need to join the inner lists in 'lem' to create strings for the DataFrame
    processed_text_strings = [' '.join(text_entry_sentences) for text_entry_sentences in lem]
    df = pd.DataFrame({'processed_text': processed_text_strings})


encoded_text_list = []
for text_entry_sentences in lem:
  # Join the sentences back into a single string for encoding
  text_entry_string = ' '.join(text_entry_sentences)
  # Fit and transform the text entry
  encoded_text_list.append(le.fit_transform([text_entry_string])[0]) # Fit and transform needs a list as input, and returns an array, so we take the first element

# Check if the length of the encoded list matches the DataFrame index length
if len(encoded_text_list) == len(df.index):
    df['encoded_text'] = encoded_text_list
    display(df.head())
else:
    print(f"Error: Length of encoded text list ({len(encoded_text_list)}) does not match DataFrame index length ({len(df.index)}).")
    print("Skipping assignment to 'encoded_text' column.")
df.shape
###classification
import pandas as pd

print("Unique values and their counts in 'encoded_text':")
# Since df is now a numpy array, use pandas Series for value_counts
print(pd.Series(df).value_counts())
print("\nFirst few rows of the encoded text array:")
# Display the first few elements of the numpy array
print(df[:5])
# X will be the features (encoded text)
X = df
print(X)

# y will be the target variable
y = text_data.target[:len(X)] # Ensure y has the same length as X
from sklearn.model_selection import train_test_split
import numpy as np # Import numpy

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape x_train and x_test to be 2D arrays with a single column
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
###probability
def prior_prob(y_train, label):
  m = y_train.shape[0]
  s = np.sum(y_train == label)
  return m/s


def cond_prob(x_train,y_train,feature_col,feature_val,label):
  x_filtered = x_train[y_train==label]
  numerator = np.sum(x_filtered[:,feature_col]==feature_val)
  denominator = x_filtered.shape[0]
  return float(numerator/denominator)
def predict(x_train,y_train,X_test):
  classes = np.unique(y_train)
  n_features = x_train.shape[1]
  posterior_prob = []

  for label in classes:
    likelihood = 1.0
    for feature in range(n_features):
      cond = cond_prob(x_train,y_train,feature,x_test[feature],label)
      likelihood = likelihood * cond
    prior = prior_prob(y_train,label)
    posterior = likelihood * prior

    posterior_prob.append(posterior)
  pred = np.argmax(posterior_prob)
  return pred

def accuracy(x_train,y_train,X_test,y_test):
  pred = []
  for i in range(x_test.shape[0]):
    p = predict(x_train,y_train,X_test[i])
    pred.append(p)
  pred = np.array(pred)
  accuracy = np.sum(pred == y_test)/y_test.shape[0]
  return accuracy
acc = accuracy(x_train,y_train,X_test,y_test)
acc*100
