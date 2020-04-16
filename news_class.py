import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


nltk.download('stopwords')

titles = []
categories = []
classes = ['Business & Finance', 'Health Care', 'Science & Health', 'Politics & Policy', 'Criminal Justice']
with open('newsdataset.tsv','r', encoding='utf-8') as tsv:
    count = 0
    for line in tsv:
        a = line.strip().split('\t')[:3]
        if a[2] in classes:
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
## make api for this too

##Data Pre Processing
# vectorize data using BOW

tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopworsds.words("english")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize(), stop_words=stop_words)

vectorizer.fit(iter(title_train))
Xtrain = vectorizer.transform(iter(title_train))
Xdev = vectorizer.transform(iter(title_dev))
Xtest = vectorizer.transform(iter(title_test))

encoder = LabelEncoder()
encoder.fit(category_train)
Ytrain = encoder.transform(category_train)
Ydev = encoder.transform(category_dev)
Ytest = encoder.transform(category_test)

##Use example for vectorizer

##Feature reduction
