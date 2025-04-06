###   微信降噪   ###
##   C:\Users\LENOVO\Desktop\微信群聚\god_ouselves.csv
##   C:\Users\LENOVO\Desktop\微信群聚\244210.csv
##   

## current Error:not precise K-means


import csv
import sys
import time
import math
import chardet
import torch
import jieba
from tqdm import tqdm
import random as rd
import numpy as np


from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer,util
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

print("already imported,wait for model")

# 加载预训练的 Sentence-BERT 模型
# 模拟模型加载过程
def load_model_with_progress(model_name):
    print(f"Loading model: {model_name}")
    with tqdm(total=100, desc="Loading", unit="%") as pbar:
        model = SentenceTransformer(model_name)
        pbar.update(100)  # 更新进度条到 100%
    # nothing but a good looking
    return model

# 使用进度条加载模型
model_Bert = load_model_with_progress("paraphrase-MiniLM-L6-v2").to("cuda")

print("model_SBERT injected")

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

print("model_BertTokenizer injected")

# -I love Midwest Emo
# -Loser

list_data = []
encode_type = {'encoding':"GB2312"}

def get_data(location):
    with open(location,"rb") as file:
        encode_type = chardet.detect(file.read())
    with open(location,"r",encoding='gbk',errors='ignore') as file:
        lines = csv.DictReader(file)
        for line in lines:
            if "<msg>" in line['StrContent']:
                continue
            data = {}
            data['nickname'] = line['NickName']
            data['content'] = line['StrContent']
            data['time'] = line['CreateTime']
            list_data.append(data)

class message:

    def __init__(self,list_data,count_message):
        self.list_data = list_data
        self.count = count_message
    
    def split_words(self):
        for i in range(len(self.list_data)):
            self.list_data[i]['characters'] = list(self.list_data[i]['content'])

    def frame_size(self):
        gama = 50
        time_gap = 300
        return gama

    def one_hot_encode(self,word_list, vocabulary):
        encoding = []
        for word in word_list:
            if word in vocabulary:
                vector = [0] * len(vocabulary)  # 初始化全 0 向量
                index = vocabulary.index(word)  # 找到字在词汇表中的索引
                vector[index] = 1  # 将对应位置设为 1
                encoding.append(vector)
            else:
                # 如果字不在词汇表中，设为全 0 向量
                encoding.append([0] * len(vocabulary))
        
        return np.array(encoding)

    def Similarity_compare(self,sentence1,sentence2):
            vocabulary = list(set(sentence2['characters']+sentence1['characters']))
            vec1 = self.one_hot_encode(sentence1['characters'],vocabulary)
            vec2 = self.one_hot_encode(sentence2['characters'],vocabulary)
            norm_vec1 = np.sum(vec1, axis=0)
            norm_vec2 = np.sum(vec2, axis=0)
            dot_product = np.dot(norm_vec1, norm_vec2)
            norm_vec1 = np.linalg.norm(norm_vec1)
            norm_vec2 = np.linalg.norm(norm_vec2)
            similarity = dot_product / (norm_vec1 * norm_vec2) if norm_vec1!= 0 and norm_vec2!= 0 else 0
            # print(f"Similarity betweem {sentence2['content']} and {sentence2['content']} is {similarity}\n")
            # time.sleep(0.5)
            return similarity
    
    def mass_remove(self):
        index = 0
        while index < len(self.list_data):
            sub_list = self.list_data[index:index + 30]
            i = 0
            while i < len(sub_list):  # 遍历子列表
                initial_point = sub_list[i]  # 获取初始点
                j = i + 1
                while j < len(sub_list):  # 遍历初始点后面的元素
                    if self.Similarity_compare(initial_point, sub_list[j]) > 0.85:
                        # print(f"the sentence {initial_point['content']} has been removed with the sentence {sub_list[j]['content']}\n")
                        # time.sleep(0.25)
                        self.list_data.pop(index + j)
                        sub_list.pop(j)
                    else:
                        j += 1  # 如果不满足条件，继续检查下一个元素
                i += 1  # 移动到下一个初始点
            index += 30

    def meaning_replace():
        
        raise(NotImplementedError)
    
    def Association_calculate(self,sentence1,sentence2):
        inputs = tokenizer(sentence1, sentence2, return_tensors='pt', max_length=128, truncation=True)
        tokens = tokenizer.tokenize(sentence1) + tokenizer.tokenize(sentence2)
        if len(tokens) > 128:
            return 0
        # 3. 进行 NSP 预测
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # 获取预测结果
            probs = torch.softmax(logits, dim=1)
            # 判断是否连贯
            coherent_prob = probs[0][1].item()  # 0 到 1 之间的值
            return coherent_prob
        
            # 计算是连贯还是不连贯
            # is_next = torch.argmax(probs, dim=1).item()  # 0 表示不连贯，1 表示连贯

        raise(NotImplementedError)
    
    def length_calculate(self,sentence):
        count_words = len(sentence)
        return math.exp(-0.1 * (count_words - 1))
        raise(NotImplementedError)
    
    def X_calculate(self,sentence1,sentence2):
        if sentence1['nickname'] == sentence2['nickname']:
            return 1
        else:
            return 0
        raise(NotImplementedError)
    
    def Distance_calculate(self,index1,index2):
        return ( 1 - (index2 - index1) / 30 )
        raise(NotImplementedError)
    
    def sentence_combine(self,Association_valve,w1,w2,w3,w4):
        frame_len = self.frame_size()
        index = 0
        while index < len(self.list_data) - 30:
            print("move to next point")
            sub_list = self.list_data[index:index + 30]
            innitial_point = sub_list[0]
            j = 1
            while j < len(sub_list):
                if w1*self.Association_calculate(innitial_point['content'],sub_list[j]['content']) + w2*self.length_calculate(sub_list[j]['content']) + w3 * self.X_calculate(innitial_point,sub_list[j]) + w4 * self.Distance_calculate(index,index + j) > Association_valve:
                    if sub_list[j]['content'] != '':
                        # time.sleep(0.5)
                        print(f"combine {self.list_data[index]['content']} with {self.list_data[index + j]['content']}")
                    self.list_data[index]['content'] += ( ';' + self.list_data[index + j]['content'])
                    self.list_data.pop(index + j)
                    sub_list.pop(j)
                else:
                    j += 1
            index += 1
        return 
        raise(NotImplementedError)
    
    def S_Bert_(self):
        for item in self.list_data:
            sentence_embeddings = model_Bert.encode(item['content'])
            item['Sb_vec'] = sentence_embeddings

    def sentence_similarity(self,sentence1,sentence2):
        return (1 - cosine(sentence1['Sb_vec'] , sentence2['Sb_vec']))
        raise(NotImplementedError)
    
    def K_means(self):
        # length_vec = len(self.list_data[0]['Sb_vec'])
        init_centroids = np.zeros_like(self.list_data[0]['Sb_vec'])
        # init_centroids = np.empty((0) * length_vec)
        sum_cluster = 0
        for index in range(len(list_data)-1):
            if self.sentence_similarity(list_data[index],list_data[index+1]) > 0.55:
                init_centroids = np.append(init_centroids,list_data[index+1]['Sb_vec'],axis=0)
                sum_cluster += 1
        # init_centroids dimension Error

        # 聚类可以是sqrt或者log
        cluster_num = math.log(len(self.list_data)) / math.log(2)
        # cluster_num = math.sqrt(len(self.list_data))
        
        cluster_num = math.ceil(cluster_num)
        clustering = SentenceClustering(self.list_data,n_clusters=cluster_num,init_centroids=init_centroids)
        clustered_data , clustered_labels = clustering.cluster_sentences()
        cluster_labels_after = list(set(clustered_labels))
        # 去重
        with open("D:\Vscode1\.venv\AI_introduction\lab\output2.txt","w",encoding='utf-8',errors='ignore') as file:
            for label in cluster_labels_after:
                print(f"lable:{label} \n")
                file.write(f"lable:{label} \n")
                for item in clustered_data[label]:
                    print(item['content'])
                    file.write(item['content'] + '\n')
        self.summary(clustered_data , clustered_labels)
        
                    
        return
        raise(NotImplementedError)

    def DBSCAN(self):

        raise(NotImplementedError)
    

    def test(self):
        for data in self.list_data:
            print(data)

    def summary(self , clustered_data , clustered_label):
        used_label = []
        clustered_labels = clustered_label.tolist()
        for label in clustered_labels:
            if label in used_label:
                clustered_labels.pop(label)
                continue
            used_label.append(label)
            print(f"lable:{label}")
            sentence = ""
            for item in clustered_data[label]:
                sentence += item['content']
            words = " ".join(jieba.cut(sentence))  # 使用 jieba 分词，并用空格连接
            # 计算 TF-IDF
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([words])  # 将字符串转换为 TF-IDF 矩阵
            except ValueError:
                print("this data is purely stopwords,which is nonsense")
                continue
            # 获取关键词
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]  # 获取 TF-IDF 值

            # 按 TF-IDF 值排序，提取权重最高的关键词
            sorted_indices = tfidf_scores.argsort()[::-1]  # 从高到低排序
            top_n = 3  # 提取前 5 个关键词
            if  len(sorted_indices) <=  3:
                print("孤立数据点：")
                print(f"{clustered_data['content']}")
            else:
                print("提取的关键词：")
                for idx in sorted_indices[:top_n]:
                    print(f"{feature_names[idx]}: {tfidf_scores[idx]}")
            clustered_labels.pop(label)
                
class SentenceClustering:
    def __init__(self, list_data, n_clusters,init_centroids=None):
        init_centroids = None
        self.list_data = list_data
        self.n_clusters = n_clusters
        self.init_centroids = init_centroids

        # 初始化 KMeans
        if init_centroids is not None:
            self.kmeans = KMeans(n_clusters=n_clusters, init=init_centroids, n_init=1)
        else:
            self.kmeans = KMeans(n_clusters=n_clusters)


    def cluster_sentences(self):
        """
        对字典列表中的句子进行 K - means 聚类
        :return: 聚类后的字典列表，每个字典新增 'cluster_label' 键，表示所属簇的标签
        """
        # 提取所有 Sentence - BERT 向量

        sb_vectors = np.array([np.array(item['Sb_vec']) for item in self.list_data])

        # 检查 sb_vectors 的形状
        print(sb_vectors.shape)  # 应该是 (n_samples, 384)

        # 如果只有一个样本，手动 reshape
        if sb_vectors.ndim == 1:
            sb_vectors = sb_vectors.reshape(1, -1)

        # 使用 K - means 聚类
        cluster_labels = self.kmeans.fit_predict(sb_vectors)
        # 将聚类标签添加到字典中
        for item, label in zip(self.list_data, cluster_labels):
            item['cluster_label'] = label
            item['cluster_type'] = type.__dictoffset__
        
        grouped_data = defaultdict(list)
        for label, item in zip(cluster_labels,self.list_data):
            grouped_data[label].append(item)

        return grouped_data,cluster_labels

def main():
    location = input("what's location?")
    get_data(location)

    Wechat_message = message(list_data,len(list_data))
    
    Wechat_message.test()

    Wechat_message.split_words()


    print('wait for mass_remove')
    time.sleep(0.5)
    # Wechat_message.mass_remove()

    print('already mass_removed')
    ##Wechat_message.test()
    # Wechat_message.meaning_replace()'
    print('wait for combine')

    start_time = time.time()
    threshold = 0.63
    w1 = 0.75
    w2 = 0.2
    w3 = 0
    w4 = 0.05
    # 设置参数
    Wechat_message.sentence_combine(threshold,w1,w2,w3,w4)

    print("already combined")
    end_time = time.time()
    elapsed_time = start_time - end_time
    print(f"elapsed time for combine is {elapsed_time}")

    print("wait for S_bert transition")
    Wechat_message.S_Bert_()

    # path = input("K-means(1) or DBSCAN(2)")
    Wechat_message.K_means() ## 兼summary

    # Wechat_message.DBSCAN()
    if 1==0 :
        print("Invalid input🤯")
        raise(ValueError)


## if __name__ == '__main__':
main()


