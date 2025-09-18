

# pip3 install requests
# pip3 install bs4
# pip3 install html5lib
import requests
from bs4 import BeautifulSoup
url="https://www.codewithharry.com"
r=requests.get(url)
htmlcontent=r.content
print(htmlcontent)
soup=BeautifulSoup(htmlcontent,'html.parser')
print(soup.prettify)
title=soup.title
print(title)
paras=soup.find_all('p')
print(paras)
anchors=soup.find_all('a')
print(anchors)
all_links=set()
for links in anchors:
    if(links.get('href') != '#'):
        link="https://www.codewithharry.com"+ links.get('href')
        all_links.add(link)
        print(link)
print(soup.find('p'))
print(soup.find('p')['class'])
print(soup.find_all('p',class_="lead"))
print(soup.find('p').get_text())


