import os
import re
import random

def split(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

   


triples=[]

with open("data/FB15k-237/all_triple.txt", "r")as f1:
        for line in f1.readlines():
            line = line.strip('\n')
            triples.append(line)  
              

test_valid,train = split(triples,shuffle=True,ratio=0.3)

print(str(len(test_valid)))
print(str(len(train)))

test,valid = split(test_valid,shuffle=True,ratio=0.5)

print(str(len(test)))
print(str(len(valid)))   


with open("data/FB15k-237/random_split_data/train_0.7.txt", "w")as f1:
    for l in train:
        f1.write(l)
        f1.write('\n')

with open("data/FB15k-237/random_split_data/test_0.15.txt", "w")as f2:
    for l in test:
        f2.write(l)
        f2.write('\n')

with open("data/FB15k-237/random_split_data/valid_0.15.txt", "w")as f3:
    for l in valid:
        f3.write(l)
        f3.write('\n')