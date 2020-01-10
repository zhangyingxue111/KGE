import os
import re

type = []

with open("data/FB15k-237/entity2type.txt", "r") as f:
    # with open("data/FB15k-237/type2id.txt", "r+") as f2:
        for line in f.readlines():
            # print(line)
            line = line.strip('\n')
            data = line.split(r" +")
            # print(data)
            for string in data:
                sub_str = string.split('\t')
                # print(sub_str)
            if sub_str:
                count=1
                for sub_sub_str in sub_str:
                    if count==1:
                        count+=1
                        continue
                    elif count>=1:
                        # print(sub_sub_str)
                        type.append(sub_sub_str)
                        # f2.write(sub_sub_str)
                        # f2.write('\n')

# print(type)

def getNonRepeatList(data):
    new_data = []
    for i in range(len(data)):
        if data[i] not in new_data:
            new_data.append(data[i])
    return new_data


new_type=getNonRepeatList(type)

print(len(new_type))

###############################----------type2id
# with open("data/FB15k-237/type2id.txt", "w") as f2:
#     id = 0
#     for sub_type in new_type:
#         f2.write(sub_type)
#         f2.write('	')
#         f2.write(str(id))
#         id+=1
#         # print(id)
#         f2.write('\n')

import numpy as np

#############################------------method1----------随机初始化
type_vec=[]

import random
def random_decode():
    for i in range(len(new_type)):
        now_vec=[]
        for j in range(100):
            r=random.random()#0-1之间抽样随机数
            num = '%3f'%r
            now_vec.append(num)
        type_vec.append(now_vec)

random_decode()
print(len(type_vec))

###############################----------type2vec
with open("data/FB15k-237/type2vec.txt", "w") as f3:
    for sub_type in type_vec:
        for vec in sub_type:
            f3.write(str(vec))
            f3.write('	')
        f3.write('\n')





