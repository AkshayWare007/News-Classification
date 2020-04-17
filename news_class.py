import numpy as np
import copy
import re
import nltk
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


nltk.download('stopwords')

titles = []
categories = []

with open('newsdataset.tsv','r', encoding='utf8') as tsv:
    count = 0
    for line in tsv:
        a = line.strip().split('\t')[:3]
        if a[2] in ['Business & Finance', 'Health Care', 'Science & Health', 'Politics & Policy', 'Criminal Justice']:
            title = a[0].lower()
            title = re.sub('\s\W',' ', title)
            title = re.sub('\W\s',' ',title)
            titles.append(title)
            categories.append(a[2])

print(titles)
print(len(titles))
print("Titles-\n", "\n".join(titles[:5]))

print(categories)
print("\nCategories-\n", "\n".join(categories[:5]))

title_train, title_test, category_train, category_test = train_test_split(titles, categories)
title_train, title_dev, category_train, category_dev = train_test_split(title_train, category_train)

print("Training: ",len(title_train))
print("Developement: ",len(title_dev),)
print("Testing: ",len(title_test))

##Visualization of data
text = " ".join(title_train)
wordcloud = WordCloud().generate(text)
plt.figure()
plt.subplots(figsize=(20,12))
wordcloud = WordCloud(
    background_color="white",
    max_words=len(text),
    max_font_size=40,
    relative_scaling=.5).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
## make api for this too

##Data Pre Processing
# vectorize data using BOW

tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopwords.words("english")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)

vectorizer.fit(iter(title_train))
print(vectorizer.vocabulary_)
Xtrain = vectorizer.transform(iter(title_train))
print(Xtrain.shape)
print(Xtrain.toarray())
Xdev = vectorizer.transform(iter(title_dev))
Xtest = vectorizer.transform(iter(title_test))

encoder = LabelEncoder()
encoder.fit(category_train)
Ytrain = encoder.transform(category_train)
Ydev = encoder.transform(category_dev)
Ytest = encoder.transform(category_test)

##Use example for vectorizer
reverse_vocabulary = {}
vocabulary = vectorizer.vocabulary_
for word in vocabulary:
    index = vocabulary[word]
    reverse_vocabulary[index] = word

vector = vectorizer.transform(iter(['Nasa scientists are good']))
indexes = vector.indices
for i in indexes:
    print (reverse_vocabulary[i])

##Feature reduction
# Number of features before reduction
print("Before features : ", Xtrain.shape[1])
selection = VarianceThreshold(threshold=0.001)
Xtrain_copy = copy.deepcopy(Xtrain)
Ytrain_copy = copy.deepcopy(Ytrain)

selection.fit(Xtrain)
Xtrain = selection.transform(Xtrain)
Xdev = selection.transform(Xdev)
Xtest = selection.transform(Xtest)

# Number of features after reduction
print("After reduction : ", Xtrain.shape[1])

## Check distribution of dataset is unifrom or not
print(Ytrain)
labels = list(set(Ytrain))
print(labels)
counts = []
for label in labels:
    print(label,"\n",Ytrain == label)
    counts.append(np.count_nonzero(Ytrain == label))
print(counts)
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.show()

# Using SMOTE we will uniformly distribute labels
sm = SMOTE(random_state=42)
Xtrain, Ytrain = sm.fit_sample(Xtrain, Ytrain)
labels = list(set(Ytrain))
counts = []
for label in labels:
    counts.append(np.count_nonzero(Ytrain == label))
plt.pie(counts, labels=labels, autopct='%1.1f%%')
plt.show()

### Multinomial Naive Bayesian
nb = MultinomialNB()
nb.fit(Xtrain, Ytrain)
pred = nb.predict(Xtest)

print(classification_report(Ytest, pred, target_names=encoder.classes_))

## Print top words in each class
nb1 = MultinomialNB()
nb1.fit(Xtrain_copy, Ytrain_copy)
coefs = nb1.coef_
target_names = encoder.classes_

for i in range(len(target_names)):
    words = []
    for j in coefs[i].argsort()[-20:]:
        words.append(reverse_vocabulary[j])
    print(target_names[i],'-', words, "\n")