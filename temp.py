#news classification - preprocessing step

#extract from json file
import json

#load the json file
newsList = []
with open('news.json', 'r') as f:
	for json_obj in f:
		news_dict = json.loads(json_obj)
		newsList.append(news_dict)

import pandas as pd
df = pd.DataFrame(newsList)

df = df.drop(columns = ["authors", "link", "date"])


#pre-process

from nltk.corpus import stopwords
import re
STOPWORDS = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
#BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
bad_symbols = ['#', '+', '-', '*', '\'']

def clean_text(text):
	text = text.lower()
	text = REPLACE_BY_SPACE_RE.sub(' ', text) #replace symbols
	text = text.replace('#','')
	text = text.replace(':','')
	text = text.replace('\'','')
	text = text.replace('?','')
	text = ''.join([i for i in text if not i.isdigit()]) #remove digits
	text = ' '.join(word for word in text.split() if word not in STOPWORDS) #remove stopwords
	return text

df.headline = df.headline.apply(clean_text)
df.short_description = df.short_description.apply(clean_text)
#save to csv file
df.to_csv('preprocess.csv')

