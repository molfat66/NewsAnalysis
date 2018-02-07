# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 00:41:02 2017

@author: mahbo
"""

import math, copy, pickle, numpy as np, scipy.sparse as sp
from sparsesvd import sparsesvd
from datetime import datetime
from tqdm import tqdm
from TopicExtraction import loadDict, loadWordRow, getM

def saveMat(wordDocMat: sp.coo_matrix) -> None:
    with open('mat.pickle','wb') as file:
        pickle.dump(wordDocMat,file)

def loadMat() -> sp.coo_matrix:
    try:
        with open('mat.pickle','rb') as file:
            wordDocMat = pickle.load(file)
    except:
        return None
    return wordDocMat

def dictToMat(wordDocDict: dict, wordRow: dict, perc=True) -> sp.coo_matrix:
    i = np.array([wordRow[word] for doc in wordDocDict.values() for word in doc[2]])
    j = np.hstack([c*np.ones(len(doc[2])) for (c,doc) in enumerate(wordDocDict.values())]).astype(int)
    vals = np.hstack([np.array(list(doc[2].values()))*(1/doc[3] if perc else 1) for doc in wordDocDict.values()])
    wordDocMat = sp.coo_matrix((vals,(i,j)),shape=(len(wordRow),len(wordDocDict)))
    return wordDocMat    
    
def getThresholds(mat: sp.csr_matrix, wordRow: dict, bnd1: float, bnd2: float, bnd3: float) -> list:
    thresh = []
    for row in mat:
        dat = row.data
        numz = mat.shape[1]-len(dat)
        holdFinal = -1
        hold = -1
        while np.sum(dat>hold+1)>=bnd1:
            hold += 1
            if np.sum(dat==hold+1)+(numz if hold==-1 else 0)<=bnd2: holdFinal = hold
        thresh.append(holdFinal if holdFinal>=bnd3 else math.inf)
    return thresh

def cluster(dat, k: int, clusterMean=None, numiter=10):
    numRows,numCols = dat.shape
    if clusterMean is None: clusterMean = dat[np.random.choice(range(numRows),k,replace=False)]
    for j in range(numiter):
        clust = [list() for x in range(k)]
        clusterTemp = np.zeros((k,numCols))
        clusterSize = np.zeros(k).astype(int)
        for i,point in tqdm(enumerate(dat)):
            if sp.issparse(point): point = np.array(point.todense()).flatten()
            best = np.argmin(np.sum((clusterMean-np.ones((k,1)).dot(point[None,:]))**2,axis=1))
            clusterTemp[best] += point
            clusterSize[best] += 1
            clust[best].append(i)
        clusterTemp[np.where(clusterSize==0)] = copy.deepcopy(clusterMean[np.where(clusterSize==0)])
        clusterSize[np.where(clusterSize==0)] = 1
        clusterMean = copy.deepcopy(clusterTemp/clusterSize[:,None].dot(np.ones((1,numCols))))
    return clust,clusterMean

def learnTopicMat(wordDocDict: dict, wordRow: dict, k: int) -> sp.csc_matrix:
    alpha = 0.4 # minimum probability of dominant topic, alpha+2delta<=0.5
    beta = 0.3 # maximum probability of non-dominant topics
    delta = 0.05 # 1-minimum probability to be considered "almost pure" document, <0.08
    rho = 0.07 # beta+rho<=(1-delta)alpha
    c0 = 1/2 # generic constant
    e0 = 1/3 # generic constant?
    p0 = 0.02 # minimum rate of occurance for anchor words?
    w0 = 1/k # generic constant?
    gamma = 1.1 #(1-2*delta)/((1+delta)*(beta+rho)) # some kind of condition number?
    m = -np.partition(-np.array(getM(wordDocDict)),10)[10]
    d = len(wordRow) # number of words in vocabulary
    s = int(len(wordDocDict)/2)+1 # number of documents in corpus
    e = min(alpha*p0/(900*c0**2*k**3*m),e0*math.sqrt(alpha*p0)*delta/640*m*math.sqrt(k))
    
    # keep the same m for all documents, just bias thresholds towards being higher
    # SPLIT DATA TO FIND THRESHOLDS
    idx = np.random.choice(range(len(wordDocDict)),len(wordDocDict),replace=False)
    A1Dict = dict(np.array(list(wordDocDict.items()))[idx[s:]])
    A2Dict = dict(np.array(list(wordDocDict.items()))[idx[:s]])
    thresholds = np.array(getThresholds(dictToMat(A1Dict,wordRow,perc=False).tocsr(),wordRow,w0*s/2,3*e*w0*s,8*math.log(20/(e*w0))))
    # THRESHOLD MATRIX
    B = dictToMat(A2Dict,wordRow,perc=False)
    B.data[np.where(B.data>thresholds[B.row])] = np.sqrt(thresholds[B.row])
    B.data[np.where(B.data<=thresholds[B.row])] = 0
    B = B.tocsc()
    # PROJECT AND CLUSTER TO INITIALIZE, THEN CLUSTER AGAIN TO GET DOMINANT TOPIC FOR EACH DOCUMENT
    ut,s,vt = sparsesvd(B,k)
    docTopics = cluster(B.T,k,cluster(vt.T.dot(np.diag(s)),k)[1].dot(ut))[0]
    # FIND CATCHWORDS FOR DOMINANT TOPICS
    A2 = dictToMat(A2Dict,wordRow).tocsr()
    bnd = int(e0*w0*s/2)
    catchWords = [list() for topic in docTopics]
    for i,word in enumerate(A2):
        maxGij = 0; idxGij = -1; distinctEnough = False
        word1 = np.array(word.todense()).flatten()
        for j,topic in enumerate(docTopics):
            Gij = -np.partition(-word1[topic],bnd-1)[bnd-1]
            if Gij<=maxGij*gamma or maxGij<=Gij*gamma: distinctEnough = False
            if Gij>maxGij:
                if Gij>maxGij*gamma: distinctEnough = True
                maxGij = Gij
                idxGij = j
        if maxGij>4*math.log(20/(e*w0))/(m*delta**2) and distinctEnough:
            catchWords[idxGij].append(i)
    # FIND ALMOST PURE DOCUMENTS FOR EACH TOPIC AND CONSTRUCT TOPIC VECTORS FROM AVERAGE
    topicDocs = [np.argpartition(-np.array(np.sum(A2[topic],axis=0)).flatten(),bnd-1)[:bnd] for topic in catchWords]
    A2 = A2.tocsc()
    M = sp.hstack([sp.csc_matrix(A2[:,docs].sum(axis=1))/len(docs) for docs in topicDocs])
    return M

if __name__=='__main__':
    wordDocDict = loadDict()
    wordRow = loadWordRow()
    k = 20
    alpha = 0.42 # minimum probability of dominant topic, alpha+2delta<=0.5
    beta = 0.25 # maximum probability of non-dominant topics
    delta = 0.07 # 1-minimum probability to be considered "almost pure" document, <0.08
    rho = 0.07 # beta+rho<=(1-delta)alpha
    c0 = 1/2 # generic constant
    e0 = 1/3 # generic constant?
    p0 = 0.02 # minimum rate of occurance for anchor words?
    w0 = 1/k # generic constant?
    gamma = 1.5 #(1-2*delta)/((1+delta)*(beta+rho)) # some kind of condition number?
    m = -np.partition(-np.array(getM(wordDocDict)),10)[10]
    d = len(wordRow) # number of words in vocabulary
    s = int(len(wordDocDict)/2)+1 # number of documents in corpus
    e = 0.1 #min(alpha*p0/(900*c0**2*k**3*m),e0*math.sqrt(alpha*p0)*delta/640*m*math.sqrt(k))
    
    # keep the same m for all documents, just bias thresholds towards being higher
    # SPLIT DATA TO FIND THRESHOLDS
    idx = np.random.choice(range(len(wordDocDict)),len(wordDocDict),replace=False)
    A1Dict = dict(np.array(list(wordDocDict.items()))[idx[s:]])
    A2Dict = dict(np.array(list(wordDocDict.items()))[idx[:s]])
    time = datetime.now()
    thresholds = np.array(getThresholds(dictToMat(A1Dict,wordRow,perc=False).tocsr(),wordRow,w0*s/10,3*e*w0*s,math.log(1/(e*w0))))
    # THRESHOLD MATRIX
    B = dictToMat(A2Dict,wordRow,perc=False)
    idx = np.where(B.data>thresholds[B.row])
    B.data = np.sqrt(thresholds[B.row])[idx]
    B.row = B.row[idx]
    B.col = B.col[idx]
    B = B.tocsc()
    print('Thresholded matrix in %s seconds' % (datetime.now()-time).seconds)
    time = datetime.now()
    # PROJECT AND CLUSTER TO INITIALIZE, THEN CLUSTER AGAIN TO GET DOMINANT TOPIC FOR EACH DOCUMENT
    ut,S,vt = sparsesvd(B,k)
    print('Projected matrix in %s seconds' % (datetime.now()-time).seconds)
    time = datetime.now()
    docTopics = cluster(B.T,k,cluster(vt.T.dot(np.diag(S)),k)[1].dot(ut),numiter=2)[0]
    print('Clustered in %s minutes, %s seconds' % (int((datetime.now()-time).seconds/60),(datetime.now()-time).seconds%60))
    time = datetime.now()
    # FIND CATCHWORDS FOR DOMINANT TOPICS
    A2 = dictToMat(A2Dict,wordRow).tocsr()
    bnd = 1#int(e0*w0*s/20)
    catchWords = [list() for topic in docTopics]
    G = np.empty((d,k))
    for i,word in enumerate(A2):
        maxGij = 0; idxGij = -1; distinctEnough = False
        word1 = np.array(word.todense()).flatten()
        for j,topic in enumerate(docTopics):
            Gij = -np.partition(-word1[topic],bnd-1)[bnd-1] if len(topic)>0 else 0
            G[i,j] = Gij
            if (Gij<=maxGij*gamma and Gij>=maxGij) or (maxGij<=Gij*gamma and maxGij>=Gij):
                distinctEnough = False
            if Gij>maxGij:
                if Gij>maxGij*gamma: distinctEnough = True
                maxGij = Gij
                idxGij = j
        if maxGij>2*math.log(20/(e*w0))/(m*delta) and distinctEnough:
            catchWords[idxGij].append(i)
    print('Found catchwords in %s minutes, %s seconds' % (int((datetime.now()-time).seconds/60),(datetime.now()-time).seconds%60))
    time = datetime.now()