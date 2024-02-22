import pandas as pd
import numpy as np
import re
from nltk import word_tokenize,pos_tag
from nltk.stem import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords


df=pd.read_excel('dialog_talk_agent.xlsx')
# print(df.head())

#fiill null value
df.ffill(axis=0,inplace=True)
# print(df.head())

def Convert(x):
    for i in x:
        a=str(i).lower()
        p=re.sub(r'[^a-z0-9]',' ',a)

# lemmantize the text 
def TextNormalization(text):
    text=str(text).lower()
    spl_char_text=re.sub(r'[^ a-z]','',text)
    tokens=word_tokenize(spl_char_text)
    lema=wordnet.WordNetLemmatizer()
    tags_list=pos_tag(tokens,tagset=None)
    lemma_words=[]
    for token,pos_token in tags_list:
        if(pos_token.startswith('V')): #verb
            pos_val='v'
        elif(pos_token.startswith('J')): #adjective
            pos_val='a'
        elif(pos_token.startswith('R')): #adverb
            pos_val='a'
        else:
            pos_val='n'  #noun

        lems_token=lema.lemmatize(token,pos_val)
        lemma_words.append(lems_token)
    
    return ' '.join(lemma_words)
        
# applying the lammantized text 
df['lemmatized_text']=df['Context'].apply(TextNormalization)

# vectorizing the text here vectorizing means getting all the words from my dataset
cv=CountVectorizer()
X=cv.fit_transform(df['lemmatized_text']).toarray()
features=cv.get_feature_names_out()
df_bow= pd.DataFrame(X,columns=features)

stop=stopwords.words('english')

query='are you busy why are you slow'

# removing unneccary words 
q=[]
a= query.split()
for i in a:
    if i in stop:
        continue
    else:
        q.append(i)
    b=" ".join(q)
    
# passing query for lemmaning text 
question_lemma=TextNormalization(b)

# transforming the query to vector array 
question_bow=cv.transform([question_lemma]).toarray()

# caluclating similarity between each df_bow to query and finding max 
cosine_value=1- pairwise_distances(df_bow,question_bow,metric='cosine')
index_value=cosine_value.argmax()

print(df['Text Response'].loc[index_value])


