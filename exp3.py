# -*- coding:utf-8 -*-
import numpy as np
from gensim import corpora, models, similarities
from pprint import pprint
import time

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import jieba
from collections import Counter
import math
import random
import openpyxl
def load_stopword():
    '''
    加载停用词表
    :return: 返回停用词的列表
    '''
    f_stop = open('cn_stopwords.txt', encoding='utf-8')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw


import os
#随机产生200个段
def DFS_file_search(dict_name):
    stack = []
    result_txt = []
    stack.append(dict_name)
    while len(stack) != 0:
        temp_name = stack.pop()
        try:
            temp_name2 = os.listdir(temp_name)
            for eve in temp_name2:
                stack.append(temp_name + "\\" + eve)  # 维持绝对路径的表达
        except NotADirectoryError:
            result_txt.append(temp_name)
    return result_txt
path_list = DFS_file_search(r".\全集")
# path_list 为包含所有小说文件的路径列表
ttext = []
for path in path_list:
    with open(path, "r", encoding="ANSI") as file:
        text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
        ttext.append(text)
String_str=["\u3002","\uff1b","\uff0c","\uff1a","\u201c","\u201d","\uff08","\uff09","\u3001","\uff1f","\u300a","\u300b"]
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    elif uchar in String_str:
        return True
    else:
        return False
def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str

ttext0=[]
for i in range(len(ttext)):
    cdcd=[]
    for j in range(len(ttext[i])):
        cdcd.append(format_str(ttext[i][j]))
    ttext0.append(cdcd)


# print(format_str(ttext[0]))
stop_words = load_stopword()
import jieba
quanji_x=[]
quanji_y=[]
we10=0
for we in range(len(ttext0)):
    jj=0
    for we1 in range(len(ttext0[we])):#每一篇
        temp=[]
        if len(jieba.lcut(ttext0[we][we1])) >=40 and we1>we10+40:#某一段大于10
            for juju in range(40):
                cd_100=[]
                cd_99=jieba.lcut(ttext0[we][we1+juju])
                for dd in cd_99:
                    if dd not in stop_words:
                        cd_100.append(dd)
                temp+=cd_100
            assert len(temp)>700
            quanji_x.append(temp)
            quanji_y.append(we)
            jj+=1
        if jj>30:#大于20就下一篇
            we10 = we1
            break
# assert len(quanji_x)>200
print(len(quanji_x))
print(quanji_y)
changdu=[]
for hi in quanji_x:
    changdu.append(len(hi))
print('max',max(changdu))
print('min',min(changdu))
print('min',sum(changdu)/len(changdu))
time.sleep(300)












if __name__ == '__main__':

    print('1.初始化停止词列表 ------')
    # 开始的时间
    t_start = time.time()
    # 加载停用词表
    stop_words = load_stopword()
    print(stop_words)

    print('2.开始读入语料数据 ------ ')

    # 读入语料库
    f = open('preprocess.txt', "r",encoding='utf-8')
    texts=[]
    for line in f:
        cd=jieba.lcut(line)
        cddd=[]
        for dd in cd:
            if dd not in stop_words:
                cddd.append(dd)
        texts.append(cddd)
    # texts = [jieba.lcut(line) for line in f]
    print('读入语料数据完成，用时%.3f秒' % (time.time() - t_start))
    f.close()
    M = len(texts)
    print('文本数目：%d个' % M)

    print('3.正在建立词典 ------')
    # 建立字典
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)

    print('4.正在计算文本向量 ------')
    # 转换文本数据为索引，并计数
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpus1 = [dictionary.doc2bow(text1) for text1 in quanji_x]

    print('len(corpus1',len(corpus1))
    print('len(corpus）',len(corpus))

    print('5.正在计算文档TF-IDF ------')
    t_start = time.time()
    # 计算tf-idf值
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    corpus_tfidf1 = models.TfidfModel(corpus1)[corpus1]
    print('corpus_tfidf1 ', len(corpus_tfidf1) )
    print('建立文档TF-IDF完成，用时%.3f秒' % (time.time() - t_start))

    print('6.LDA模型拟合推断 ------')
    # 训练模型
    num_topics = 15
    t_start = time.time()
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                      alpha=0.01, eta=0.01, minimum_probability=0.001,
                      update_every=1, chunksize=3000, passes=20)
    print('LDA模型完成，训练时间为\t%.3f秒' % (time.time() - t_start))

    # 随机打印某10个文档的主题
    # 摘取200个段
    num_show_topic = 20  # 每个文档显示前几个主题
    print('7.结果：10个文档的主题分布：--')

    print(corpus_tfidf)


    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    doc_topics1 = lda.get_document_topics(corpus_tfidf1)  # 所有文档的主题分布

    wb = openpyxl.load_workbook("Test.xlsx")
    sheet = wb['Sheet1']

    print('len(doc_topics1)',len(doc_topics1))
    print('len(doc_topics1)',len(quanji_y))

    for p in range(len(doc_topics1)):
        topic = np.array(doc_topics1[p])
        topic_distribute = np.array(topic[:, 1])
        # topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
        topic_idx = topic_distribute[:-num_show_topic - 1:-1]
        print('第%d个文档的前%d个主题：' % (p, num_show_topic)), topic_idx
        # print(topic_distribute[topic_idx])
        print(topic_distribute)

        cdd=[quanji_y[p]]
        # cdd+=list(topic_distribute[topic_idx])
        cdd+=list(topic_distribute)
        sheet.append(cdd)
    wb.save("Test.xlsx")


    # idx = np.arange(M)
    # np.random.shuffle(idx)
    # idx = idx[:10]
    # for i in idx:
    #     topic = np.array(doc_topics[i])
    #     topic_distribute = np.array(topic[:, 1])
    #     # print topic_distribute
    #     topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
    #     print('第%d个文档的前%d个主题：' % (i, num_show_topic)), topic_idx
    #     print(topic_distribute[topic_idx])

    num_show_term = 10  # 每个主题显示几个词
    print('8.结果：每个主题的词分布：--')
    for topic_id in range(num_topics):
        print('主题#%d：\t' % topic_id)
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        print('词：\t', )
        for t in term_id:
            print(dictionary.id2token[t], )
        print('\n概率：\t', term_distribute[:, 1])



