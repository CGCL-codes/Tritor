import os
import csv
import numpy as np
import time
import json
from itertools import islice


with open('feasite_dict.json', 'r', encoding='utf8') as fp:
    groupindex = json.load(fp)


def mkdata(file, datas):
    with open(file, 'r') as f:
        a = f.read()
        data = a.split(',')[:-1]
        id = file.split('/')[-1].split('.txt')[0]
        datas[id] = data


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


def jaccard_sim(group1, group2):
    intersection = 0
    union = 0
    triads_num1 = {}
    triads_num2 = {}
    for triad1 in group1:
        triads_num1[triad1] = triads_num1.get(triad1, 0) + 1
    for triad2 in group2:
        triads_num2[triad2] = triads_num2.get(triad2, 0) + 1

    for triad in list(set(group1).union(set(group2))):
        intersection += min(triads_num1.get(triad, 0), triads_num2.get(triad, 0))
        union += max(triads_num1.get(triad, 0), triads_num2.get(triad, 0))

    # 除零处理
    sim = float(intersection) / union if union != 0 else 0
    return sim


def group_division(txt):
    groups = [[] for i in range(len(groupindex))]
    for word in txt:
        words = word.split('/')
        a = groupindex[words[0]+words[1]]
        b = groupindex[words[0]+words[2]]
        c = groupindex[words[0]+words[3]]
        groups[a].append(word)
        groups[b].append(word)
        groups[c].append(word)
    return groups


def sim(pairs, filedict, outpath):
    j = 0
    features = []
    for p in pairs:
        f1 = p[0].split('.java')[0]
        f2 = p[1].split('.java')[0]
        if f1 in filedict and f2 in filedict:
            jacfea = []
            txt1 = filedict[f1]
            txt2 = filedict[f2]
            groups1 = group_division(txt1)
            groups2 = group_division(txt2)
            for i in range(len(groups1)):
                feature1 = groups1[i]
                feature2 = groups2[i]
                jacfea.append(jaccard_sim(feature1, feature2))
            name = [f1, f2]
            name.extend(jacfea)
            features.append(name)
            print(j)
            j += 1

    with open(outpath + '_jac_sim.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in features:
            writer.writerow(row)


def main(documents, pairspath, outpath):
    pairs = csv.reader(open(pairspath, 'r', encoding='gbk'))
    sim(pairs, documents, outpath)


if __name__ == '__main__':
    start1 = time.time()
    # 从文件夹中读取所有Java文件对应的文档
    javapath = './triadstxt/'
    javalist = []
    listdir(javapath, javalist)
    documents = {}

    for javafile in javalist:

        mkdata(javafile, documents)
    # 非克隆对相似性
    # main(documents, './dataset/BCB/BCB_clone.csv', 'BCB_clone')
    # main(documents, './dataset/BCB/BCB_nonclone.csv', 'BCB_nonclone')
    # main(documents, './dataset/BCB/T1.csv', 'T1')
    # main(documents, './dataset/BCB/T2.csv', 'T2')
    # main(documents, './dataset/BCB/ST3.csv', 'ST3')
    # main(documents, './dataset/BCB/MT3.csv', 'MT3')
    # main(documents, './dataset/BCB/WT3T4.csv', 'WT3T4')

    main(documents, './GCJ/GCJ_clone.csv', 'GCJ_clone')
    main(documents, './GCJ/GCJ_noclone_270000.csv', 'GCJ_nonclone')

    end1 = time.time()
    t1 = end1 - start1
    print('time:')
    print(t1)