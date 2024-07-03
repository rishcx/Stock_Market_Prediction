import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
from scipy.special import softmax
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BartTokenizer, BartForConditionalGeneration

def scraping_article(url):
    headers = {
    'User-Agent': 'Your User Agent String',
    }
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = [paragraph.text for paragraph in paragraphs]
    words = ' '.join(text).split(' ')
    article = ' '.join(words)
    return article

def find_url(keyword):
    root = "https://www.google.com/"
    search_query = keyword.replace(" ", "+")
    link = f"https://www.google.com/search?q={search_query}&tbm=nws"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(link, headers=headers)
    webpage = response.content
    soup = BeautifulSoup(webpage, 'html5lib')
    links = []
    for div_tag in soup.find_all('div', class_='Gx5Zad'):
        a_tag = div_tag.find('a')
        if a_tag:
            if 'href' in a_tag.attrs:
                href = a_tag['href']
                if href.startswith('/url?q='):
                    url = href.split('/url?q=')[1].split('&sa=')[0]
                    links.append(url)
    return links

def to_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=50
    )
    docs = text_splitter.split_text(data)
    return docs

def load_bart_model(model_name="facebook/bart-large-cnn"):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def find_news_url(keyword, start_date, end_date):
    root = "https://www.google.com/"
    search_query = keyword.replace(" ", "+")
    link = f"{root}search?q={search_query}&tbm=nws&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    response = requests.get(link, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    news_links = []

    for article in soup.select('div.SoaBEf'):
        link = article.select_one('a')
        if link and 'href' in link.attrs:
            url = link['href']
            if url.startswith('/url?q='):
                url = unquote(url.split('/url?q=')[1].split('&sa=')[0])
            news_links.append(url)

    return news_links

def summarize_text(tokenizer, model, text, max_chunk_length, summary_max_length):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_chunk_length, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=summary_max_length, min_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_article(url, model_name="facebook/bart-large-cnn"):
    data = scraping_article(url)
    chunks = to_chunks(data)
    tokenizer, model = load_bart_model(model_name)
    summaries = []
    for chunk in chunks:
        chunk_text = chunk
        summary = summarize_text(tokenizer, model, chunk_text, 3000, 800)
        summaries.append(summary)
    concatenated_summaries = " ".join(summaries)
    intermediate_chunks = [concatenated_summaries[i:i+3000] for i in range(0, len(concatenated_summaries), 3000)]
    final_summaries = []
    for intermediate_chunk in intermediate_chunks:
        final_summary = summarize_text(tokenizer, model, intermediate_chunk, 3000, 800)
        final_summaries.append(final_summary)    
    final_summary_text = " ".join(final_summaries)    
    return final_summary_text

def senti_model(model_name="cardiffnlp/twitter-roberta-base-sentiment"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def find_senti(news_texts):
    tokenizer, model = senti_model()
    encoded = tokenizer(news_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
    output = model(**encoded)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    weights = {
        'neg': -1,
        'neu': 0,
        'pos': 1
    }
    probabilities = {
        'neg': scores[0],
        'neu': scores[1],
        'pos': scores[2]
    }
    compound_score = sum(probabilities[label] * weights[label] for label in probabilities)
    senti_dict = {
        'neg': scores[0],
        'neu': scores[1],
        'pos': scores[2],
        'polarity': compound_score        
    }
    return senti_dict

def extract_features(summary):
    sentiment_scores = find_senti(summary)
    features = {
        'compound_sentiment_score': sentiment_scores['polarity'],  
        'negative_sentiment_score': sentiment_scores['neg'],
        'neutral_sentiment_score': sentiment_scores['neu'],
        'positive_sentiment_score': sentiment_scores['pos']
    }    
    return features

def analyze_stock(stock_name):
    urls = find_url(stock_name)
    summaries = []
    for i in range(5):
        summary = summarize_article(urls[i])
        summaries.append(summary)

    all_scores = []
    for i in range(5):
        scores = extract_features(summaries[i])
        all_scores.append(scores)

    avg_score = {}
    avg_comp = 0
    avg_pos = 0
    avg_neg = 0
    avg_neu = 0
    for i in range(5):
        avg_comp += all_scores[i]["compound_sentiment_score"]
        avg_neg += all_scores[i]["negative_sentiment_score"]
        avg_neu += all_scores[i]["neutral_sentiment_score"]
        avg_pos += all_scores[i]["positive_sentiment_score"]
    avg_score["avg_compound_score"] = avg_comp / 5
    avg_score["avg_negative_score"] = avg_neg / 5
    avg_score["avg_neutral_score"] = avg_neu / 5
    avg_score["avg_positive_score"] = avg_pos / 5

    result = ""
    if avg_score["avg_compound_score"] > 0:
        result = "The stock will go up based on the following articles:\n"
    else:
        result = "The stock will go down based on the following articles:\n"

    for url in urls[:5]:
        result += f"{url}\n"

    return result, avg_score

if __name__ == "__main__":
    stock_name = input("Enter Stock Name: ")
    result, avg_score = analyze_stock(stock_name)
    print(result)
    print("Average Sentiment Scores:")
    for key, value in avg_score.items():
        print(f"{key}: {value}")