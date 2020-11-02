import requests
import re
import time
from bs4 import BeautifulSoup


def get_page_urls(url):
    final_urls = []
    if ('en.ethereumworldnews.com' in url):
        response = requests.get(url).text
        response_urls = re.findall(r'(?<=url\": \").*(?=\/)', response)
        for resp_url in response_urls:
            if ('category' in resp_url or 'wp-content' in resp_url or '-' not in resp_url):
                continue
            final_urls.append(resp_url)
    return final_urls


def get_page_content(url):
    text_from_tags = []
    if ('en.ethereumworldnews.com' in url):
        response = requests.get(url).text
        soup = BeautifulSoup(response,'html.parser')
        result = soup.find_all("div", {"class":"entry-content body-color clearfix link-color-wrap"},'html.parser')[0]
        soup = BeautifulSoup(str(result),'html.parser')
        tags = soup.find_all(['p','li'])
        for tag in tags:
            text_from_tags.append(tag.text.strip())
    return text_from_tags


def write_to_file(path,sentence_list):
    with open(path,'w') as file:
        file.write('')
    file = open(path,'a')
    for sentence in sentence_list:
        try:
            file.write(sentence + '\n')
        except Exception as e:
            with open('error_logs.txt','a') as f:
                f.write(time.asctime(time.localtime(time.time()))+ ':\t' + str(e) + '-'+ path +'\n')
                print(str(e))
    file.close()


def get_url_list(file_path):
    file = open(file_path,'r')
    text = file.read()
    return text.split('\n')


def get_news(data_path, main_url,url_list_path):
    urls = get_page_urls(url = main_url)
    url_list = get_url_list(url_list_path)
    for url in urls:
        if (url not in url_list):
            write_to_file(data_path + url.split('/')[3] + '.txt', get_page_content(url))
            with open(url_list_path,'a') as file:
                file.write(url + '\n')


if __name__ == '__main__':
    last_checked_time = time.time()
    url_list_path = 'url_list.txt'
    main_url = 'https://en.ethereumworldnews.com/category/news/latest-ethereum-eth-news/'
    while True:
        if (time.time() - last_checked_time > 300):
            try:
                get_news('data/',main_url,url_list_path)
            except Exception as e:
                print(str(e))
            last_checked_time = time.time()
