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
                del sub_str[0]
                # print(sub_str)
                # print('\n')
                type.append(sub_str)
            # if sub_str:

            
###############################----------type2vec
with open("data/FB15k-237/type_re/types.txt", "w") as f2:
    for sub_type in type:
        f2.write(str(sub_type))    
        f2.write('\n')