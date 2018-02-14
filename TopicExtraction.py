# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:20:21 2017

@author: mahbo
"""
import newspaper, requests, csv, pickle, winsound, warnings, random, socket, wmi, numpy as np, scipy.sparse as sp
from collections import namedtuple
from tqdm import tqdm
from datetime import datetime
from time import sleep
from bs4 import BeautifulSoup
from newspaper.article import ArticleException
from newspaper.source import SourceException

document = namedtuple('document', 'title authors words length city state paper')

def isConnected():
    try:
        socket.create_connection(('www.google.com',80))
        return True
    except OSError:
        pass
    return False

def enoughCharge(overRide=False):
    try:
        c = wmi.WMI()
        t = wmi.WMI(moniker = "//./root/wmi")
    
        full = t.ExecQuery('Select * from BatteryFullChargedCapacity')[0].FullChargedCapacity
        bat = t.ExecQuery('Select * from BatteryStatus where Voltage > 0')[0]
        curr = bat.RemainingCapacity
        charging = bat.Charging
        critical = bat.Critical
        perc = round(100*curr/full,2)
        if not critical and (charging or perc>15 or overRide): return True
    except:
        pass
    return False

def getM(wordDocDict) -> list:
    return [document[3] for document in wordDocDict.values()]

def saveWordRow(wordRow: set) -> None:
    with open('wordRow.pickle','wb') as file:
        pickle.dump(wordRow,file)

def saveDict(wordDocDict: dict) -> None:
    with open('dict.pickle','wb') as file:
        pickle.dump(wordDocDict,file)
        
def loadWordRow() -> set:
    try:
        with open('wordRow.pickle','rb') as file:
            wordRow = pickle.load(file)
    except:
        return None
    return wordRow

def loadDict() -> dict:
    try:
        with open('dict.pickle','rb') as file:
            wordDocDict = pickle.load(file)
    except:
        return None
    return wordDocDict
    
def scrapeSites() -> None:
    with open('Sites.csv', 'w',newline='') as file:
        c = csv.writer(file,delimiter=',')
        US = ['ak','al','ar','az','ca','co','ct','dc','de','fl','ga','hi','ia',\
              'id','il','in','ks','ky','la','ma','md','me','mi','mn','mo','ms',\
              'mt','nc','nd','ne','nh','nj','nm','nv','ny','oh','ok','or','pa',\
              'ri','sc','sd','tn','tx','ut','va','vt','wa','wi','wv','wy']
        for state in US:
            try:
                html_doc = requests.get('http://www.usnpl.com/'+state+'news.php')
            except:
                print('Skipping '+state+'...')
                continue
            soup = BeautifulSoup(html_doc.content,'html.parser')
            for html in soup.find_all('body')[0].find_all('b'):
                try:
                    link = html.next_sibling.next_sibling
                    if 'href' in link.attrs and html.string!='Click':
                        c.writerow([link.get('href'),html.string,state])
                except:
                    pass

def constructDict(wordDocDict=None,wordRow=None,build=False,myfile=None):
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    substitute = namedtuple('substitute', 'old new')
    site = namedtuple('site','url city state')
    subList = [substitute('\n',' ')]
    with open('Text_Clean.csv','r') as csvfile:
        filereader = csv.reader(csvfile,delimiter=',')
        for pair in filereader:
            subList.append(substitute(old=pair[0],new=pair[1]))
    with open('Common_Words.csv','r') as csvfile:
        filereader = csv.reader(csvfile,delimiter=',')
        commonWords = [word[0] for word in filereader]
    with open('Sites.csv','r') as csvfile:
        filereader = csv.reader(csvfile,delimiter=',')
        sites = [site(line[0],line[1],line[2]) for line in filereader]
    with open('Bad_Sites.csv','r') as csvfile:
        filereader = csv.reader(csvfile,delimiter=',')
        badSites = [url[0] for url in filereader]
    with open('Scraped_Sites.csv','r') as csvfile:
        filereader = csv.reader(csvfile,delimiter=',')
        scrapedSites = [url[0] for url in filereader if len(url)>0]
    
    exitFlag = False
    random.shuffle(sites)
    
    if wordDocDict is None: wordDocDict = {}
    if wordRow is None: wordSet = set()
    else: wordSet = set(wordRow.keys())
    
    time = datetime.now()
    with open('Scraped_Sites.csv','a',newline='') as csvfile1:
        scrapedSiteWriter = csv.writer(csvfile1,delimiter=',')
        with open('Bad_Sites.csv','a',newline='') as csvfile2:
            badSiteWriter = csv.writer(csvfile2,delimiter=',')
            for site in tqdm([site for site in sites if site.url not in scrapedSites]):
                if site.url in badSites: continue
                if build and site.url in scrapedSites: continue
                try:
                    try:
                        paper = newspaper.build(site.url,memoize_articles=False)
                    except SourceException:
                        scrapedSiteWriter.writerow([site.url])
                        badSiteWriter.writerow([site.url])
                        continue
                    numArticle = 1
                
                    for article in paper.articles:
                        try:
                            article.download()
                            sleep(150/1000.0)
                            article.parse()
                            text = article.text.lower()
                        except KeyboardInterrupt:
                            exitFlag = True
                            break
                        except ArticleException:
                            badSiteWriter.writerow([article.url,paper.brand])
                            continue
                        except:
                            continue
                        if len(text)<=500: continue
                        
                        for sub in subList:
                            text = text.replace(sub.old,sub.new)
                        words = text.split(' ')
                        if len(words)<100: continue
                        wordCount = {word: words.count(word) for word in set(words) if word not in commonWords and not word.isnumeric()}
                        wordSet = wordSet | set(wordCount.keys())
                        wordDocDict[datetime.now().strftime('%y-%m-%d')+' | '+article.url] = (article.title,article.authors,wordCount,sum(wordCount.values()),site.city,site.state,paper.brand)
                        numArticle += 1
                    if exitFlag or not enoughCharge(overRide=True): raise KeyboardInterrupt
                    scrapedSiteWriter.writerow([site.url])
                except KeyboardInterrupt:
                    communicate('\nBroke loop at '+paper.brand,myfile)
                    break
                if (datetime.now()-time).seconds>2*60*60 or not enoughCharge() or not isConnected(): break
    wordRow = {word: i for (i,word) in enumerate(wordSet)}
    return wordDocDict, wordRow

def clearWordFromDict(word: str, wordDocDict: dict, wordRow: dict):
    [item[2].pop(word) for item in wordDocDict.values() if word in item[2].keys()]
    if word in wordRow.keys():
        row = wordRow[word]
        wordRow.pop(word)
        wordRow = {wor: i-1 if i>row else i for wor,i in wordRow.items()}
    return wordDocDict, wordRow

def clearDupes(wordDocDict: dict) -> dict:
    titles = [value[0] for value in wordDocDict.values()]
    lose = []
    for title in [title for title in set(titles) if titles.count(title)>1]:
        copies = [(item[0],item[1][2]) for item in wordDocDict.items() if item[1][0]==title]
        keep = []
        for url,dic in copies:
            if dic in keep:
                lose.append(url)
            else:
                keep.append(dic)
    [wordDocDict.pop(url) for url in lose]
    return wordDocDict
    
def communicate(msg,myfile=None):
    if myfile is not None: myfile.write('\n'+msg)
    else: print(msg)

if __name__ == '__main__':
    with open('log.txt','a') as myfile:
        communicate(str(datetime.now()),myfile)
        if enoughCharge(overRide=False):
            if isConnected():
                wordDocDict = loadDict()
                wordRow = loadWordRow()
                wordDocDict, wordRow = constructDict(wordDocDict,wordRow,build=False,myfile=myfile)
                wordDocDict = clearDupes(wordDocDict)
                saveDict(wordDocDict)
                saveWordRow(wordRow)
            else:
                communicate('No internet connection',myfile)
        else:
            communicate('Not enough battery capacity',myfile)
        communicate(str(datetime.now()),myfile)
    winsound.PlaySound("SystemExit",winsound.SND_ALIAS)