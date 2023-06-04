#!/usr/bin/env python
# coding: utf-8

# ## Web Scraping using BeautifulSoup

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

headers = {'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'}
url = 'https://www.google.com/search?q=artificial+intelligence&source=lmns&tbm=nws&bih=689&biw=1280&rlz=1C1CHBF_enIN993IN993&hl=en&sa=X&ved=2ahUKEwjQ-cvS3KT_AhWc3HMBHSH8CvEQ_AUoAnoECAEQAg'

title = []
source = []
link = []
published = []

def get_links(url):
    page = requests.get(url , headers = headers)
    soup = bs(page.content , 'html.parser')
    item_list = soup.find_all(class_="SoaBEf")
    for item in item_list:
        title.append(item.find(class_='n0jPhd ynAwRc MBeuO nDgy9d').text)
        source.append(item.span.text)
        link.append(item.find("a",class_="WlydOe")["href"])
        published.append(item.find(class_="OSrXXb rbYSKb LfVVr").span.text)
    if(soup.find("a" , id="pnnext")):
        nextp = soup.find("a" , id="pnnext")["href"]
        url = "https://www.google.com"+nextp
        get_links(url)
get_links(url)
news_db = pd.DataFrame({ 'Title' : title , 'Source' : source , 'Link' : link , 'Published' : published})


# ## Downloading Article text using Newspaper Library

from newspaper import Article

def get_article_text(link):
    try:
        article = Article(link)
        article.download()
        article.parse()
        return article.text
    except:
        return "The article is not accessible"
    
news_db['Text'] = news_db['Link'].apply(get_article_text)
news_db['Text'] = news_db['Text'].str.replace("\n\n","")
news_db


news_db.to_csv("News_DB.csv")
news_db = pd.read_csv("News_DB.csv")
news_db.drop(["Unnamed: 0"],axis=1,inplace=True)
news_db.head()


# ## Abstractive Text Summarization using OpenAI API

import os
import openai
openai_api_key="<<YOUR_API_KEY>>"

def text_summarize_abstractive(text):
    summary = openai.Completion.create(
    openai_api_key,
    model="text-davinci-003",
    prompt=text +"Tl;dr",
    temperature=0.6,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=1)
    return summary["choices"][0]["text"]

news_db['Abstractive_Summary'] = news_db['Text'].apply(text_summarize_abstractive)


# ## Extractive Text Summarization using Sumy

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer

def text_summarize_extractive(text):
    parser = PlaintextParser(text , Tokenizer("english"))
    summary =""
    LSASummarizer = Summarizer()
    num_sentences = 5
    for sentence in LSASummarizer(parser.document , num_sentences):
        summary+=(str(sentence))
    return summary

news_db["Extractive_Summary"] = news_db["Text"].apply(text_summarize_extractive)
news_db




