import streamlit as st
import pandas as pd
import numpy as np
import requests
from GoogleNews import GoogleNews
from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
import nltk
nltk.download('punkt')

st.title("News Articles Scraper and Summarizer")
st.markdown("## Hey there! You can get Extractive Summary of recent news articles quickly using this web app. You also get the link to the articles so that you can browse those that interest you.")
input_key = st.text_input("Give the input for which you want to get the news :")
#input_date = st.text_input("Give the number of days prior you wish to get :")
#input_key = "Harry Potter"

def get_article_text(link):
    try:
        link = link
        article = Article(link)
        article.download()
        #print(article)
        article.parse()
        return article.text
    except:
        return "This article is not accessible" 
    

def text_summarize_extractive(text):
    parser = PlaintextParser(text , Tokenizer("english"))
    summary =""
    LSASummarizer = Summarizer()
    num_sentences = 4
    for sentence in LSASummarizer(parser.document , num_sentences):
        summary+=(str(sentence))
    return summary

if(input_key):
    googlenews = GoogleNews("1d")
    googlenews.set_lang("en")
    googlenews.search(input_key)

    result = googlenews.results(sort=True)


    news_db = pd.DataFrame(result)
    news_db.drop(["desc","datetime","img"],axis=1,inplace=True)

    news_db['Text'] = news_db['link'].apply(get_article_text)
    news_db["Extractive_Summary"] = news_db["Text"].apply(text_summarize_extractive)
    #news_db.drop(news_db[news_db['Text'] == "This article is not accessible"].index ,inplace=True)

    st.write("Here are the 10 recently published articles on " +input_key +" with their summary!")
    st.write(news_db)
#print(news_db.iloc[1])
