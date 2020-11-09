import requests
import re
import time
from bs4 import BeautifulSoup
import datetime

def get_page_urls(url):
    final_urls = []
    if ('en.ethereumworldnews.com' in url):
        response = requests.get(url).text
        response_urls = re.findall(r'(?<=url\": \").*(?=\/)', response)
        for resp_url in response_urls:
            if ('category' in resp_url or 'wp-content' in resp_url or '-' not in resp_url):
                continue
            final_urls.append(resp_url)
    if ('blog.ethereum.org' in url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tags = soup.find_all('a')
        for tag in tags:
            url_from_tag = tag.get('href')
            if (re.search(r'\d\d\d\d/', url_from_tag)):
                final_urls.append(url + url_from_tag)
    final_urls = list(dict.fromkeys(final_urls))
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
    if ('blog.ethereum.org' in url):
        response = requests.get(url).text
        soup = BeautifulSoup(response,'html.parser')
        result = soup.find_all("article", {"class":"blog-post"},'html.parser')[0]
        soup = BeautifulSoup(str(result),'html.parser')
        tags = soup.find_all(['p'])
        for tag in tags:
            text_from_tags.append(tag.text.strip())
    return text_from_tags

def get_publishing_time(url):
    time_object = None
    if ('en.ethereumworldnews.com' in url):
        response = requests.get(url).text
        soup = BeautifulSoup(response,'html.parser')
        time_str = (soup.find_all('time')[0]).get('datetime')
        time_object = datetime.datetime.fromisoformat(time_str)
    if ('blog.ethereum.org' in url):
        split_url = url.split('/')
        time_object = datetime.datetime.strptime(split_url[4] + ' ' + split_url[5] + ' ' + split_url[6], '%Y %m %d')
    return time_object.replace(tzinfo=None)


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
                #print(str(e)) # print exception
    file.close()


def get_url_list(file_path):
    file = open(file_path,'r')
    text = file.read()
    return text.split('\n')


def get_news(data_path, main_url,url_list_path):
    urls = get_page_urls(url = main_url) # get links to all articles on a main page
    url_list = get_url_list(url_list_path) # get all links for a url_list file
    for url in urls:
        if (url not in url_list):
            print('URL: ' + url)
            file_name = ''
            if ('en.ethereumworldnews.com' in url):
                file_name = url.split('/')[3]
            if ('blog.ethereum.org' in url):
                file_name = url.split('/')[7]
            publishing_time = get_publishing_time(url)
            print(publishing_time)
            write_to_file(data_path + file_name + '_'+str(publishing_time.timestamp())+ '.txt', get_page_content(url))
            with open(url_list_path,'a') as file:
                file.write(url + '\n')

if __name__ == '__main__':

    REQUEST_TIMER = 300
    last_checked_time = 0
    url_list_path = 'url_list.txt'
    main_url_list = ['https://en.ethereumworldnews.com/category/news/latest-ethereum-eth-news/','https://blog.ethereum.org/']
    for i in range(20):
        main_url_list.append('https://en.ethereumworldnews.com/category/news/latest-ethereum-eth-news/page/' + str(i))
    while True:
        if (time.time() - last_checked_time > REQUEST_TIMER):
            for main_url in main_url_list:
                try:
                    get_news('data/articles/',main_url,url_list_path)
                except Exception as e:
                    print(str(e)) # print exception
            last_checked_time = time.time()
