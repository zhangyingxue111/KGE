import os
import re

# -------------------------------------------------------------------------------生成entity—2type_min.txt
# entity=[]

# with open("data/FB15k-237/entity2id.txt", "r")as f1:
#     for line in f1.readlines():
#         line = line.strip('\n')
#         data = line.split(r" +")
#         for string in data:
#             sub_str = string.split('\t')
#             # print(sub_str[0])
#             entity.append(sub_str[0]) 

# print(len(entity))
#14541 

# entity_type=[]
# count=0
# with open("data/FB15k-237/entity2type.txt", "r")as f2:
#     for line in f2.readlines():
#             line = line.strip('\n')
#             data = line.split(r" +")
#             for string in data:
#                 # print(string)
#                 sub_str = string.split('\t')
#                 if sub_str[0] in entity:
#                     # print(sub_str[0])
#                     entity_type.append(string)  
#                 else:
#                     count+=1
          

# print(len(entity_type))
# print(str(count))
#14533  408

# with open ("data/FB15k-237/entity2type_min.txt", "w")as f3:
#     for s in entity_type:
#         f3.write(str(s))
#         f3.write('\n')

# =====================================================================================================

# -------------------------------------------------------------------------------生成type2id_min.txt
# type = []

# with open("data/FB15k-237/entity2type_min.txt", "r") as f:
#     # with open("data/FB15k-237/type2id.txt", "r+") as f2:
#         for line in f.readlines():
#             # print(line)
#             line = line.strip('\n')
#             data = line.split(r" +")
#             # print(data)
#             for string in data:
#                 sub_str = string.split('\t')
#                 # print(sub_str)
#             if sub_str:
#                 count=1
#                 for sub_sub_str in sub_str:
#                     if count==1:
#                         count+=1
#                         continue
#                     elif count>=1:
#                         # print(sub_sub_str)
#                         type.append(sub_sub_str)
#                         # f2.write(sub_sub_str)
#                         # f2.write('\n')

# # print(type)

# def getNonRepeatList(data):
#     new_data = []
#     for i in range(len(data)):
#         if data[i] not in new_data:
#             new_data.append(data[i])
#     return new_data


# new_type=getNonRepeatList(type)

# print(len(new_type))
# # 3865

# ##############################----------type2id
# with open("data/FB15k-237/type2id_min.txt", "w") as f2:
#     id = 0
#     for sub_type in new_type:
#         f2.write(sub_type)
#         f2.write('	')
#         f2.write(str(id))
#         id+=1
#         # print(id)
#         f2.write('\n')



# ======================================================================================================




# -------------------------------------------------------------------------------生成14533个entity 对应的vec_type.txt
# 1,找出所有root_type并筛选
types=set()
dict_nums={}
with open("data/FB15k-237/type2id_min.txt", "r") as f:
    for line in f.readlines():
            # print(line)
            line = line.strip('\n')
            data = line.split(r" +")
            for string in data:
                sub_str = string.split('\t')
                sub_type=sub_str[0].split('/')
                dict_nums[sub_type[1]]=dict_nums.get(sub_type[1],0)+1
                types.add(sub_type[1])

# for key,value in dict_nums.items():
#     print(key+':'+str(value))
    

for k,v in dict_nums.items():
    if v>190 or v<5:
        types.remove(k)

print('\n')
print(len(types))
# 47

# for i in types:
#     print(i)


# 2，找出每个entity对应的root_type
dict_e_types={}
with open("data/FB15k-237/entity2type_min.txt", "r") as f3:
    for line in f3.readlines():
            # print(line)
            line = line.strip('\n')
            data = line.split(r" +")
            root_type=set()
            for string in data:
                sub_str = string.split('\t')
                # print(sub_str)
                c=1
                for sub in sub_str:
                    if c==1:
                        c+=1
                        continue
                    else:
                        r_ty=sub.split('/')
                        # print(r_ty[1])
                        if r_ty[1] in types:
                            root_type.add(r_ty[1])        
                dict_e_types[sub_str[0]]=str(root_type)
            

for k,v in dict_e_types.items():
    print("entity: "+k+" types: "+v)
    break

# 3,按照entity的向量和root_type写入vec_type.txt
# 14541
entity_old=[]
entity_vec_old=[]

with open("data/FB15k-237/entity2id.txt", "r") as f4:
    for line in f4.readlines():
        line = line.strip('\n')
        data = line.split(r" +")
        for string in data:
                sub_str = string.split('\t')
                # print(sub_str)
                if sub_str:
                    for sub_sub_str in sub_str:
                        # print(sub_sub_str)
                        entity_old.append(sub_sub_str)
                        break

with open("data/FB15k-237/entity2vec_type.txt", "r") as f5:
    for line in f5.readlines():
        line = line.strip('\n')
        entity_vec_old.append(line)
        
# 14533
entity_new=[]

with open("data/FB15k-237/entity2type_min.txt", "r")as f2:
    for line in f2.readlines():
            line = line.strip('\n')
            data = line.split(r" +")
            for string in data:
                # print(string)
                sub_str = string.split('\t')
                entity_new.append(sub_str[0])  

entity_vec_GAT=[]
with open("data/FB15k-237/after_GAT_vec/after_BAT_vec_mlp(2,2).txt", "r")as f7:
    for line in f7.readlines():
        line = line.strip('\n')
        entity_vec_GAT.append(line)
        
print("entity_new: "+entity_new[0])
print("entity_new's len : "+str(len(entity_new)))
print("entity_old: "+entity_old[0])
print("entity_old's len : "+str(len(entity_old)))
print("entity_vec_old: "+entity_vec_old[0])
print("entity_vec_old's len : "+str(len(entity_vec_old)))
print("entity_vec_GAT: "+entity_vec_GAT[0])
print("entity_vec_GAT's len : "+str(len(entity_vec_GAT)))

# entity_new: /m/0h407
# entity_new's len : 14533
# entity_old: /m/027rn
# entity_old's len : 14541
# entity_vec_old: -0.282591       -1.983983       -0.186767       -1.106942       1.036489        -0.119922       0.372494        0.003803        -0.165316       1.008080        1.821133        -0.053823       0.032137   -0.377634        0.291662        -0.493058       -0.177859       0.312003        0.192391        0.124231        -0.191677       0.108303        -0.055331       -0.161899       0.153300        0.276398        0.003765   0.178272 -0.216785       -0.138257       -0.129585       0.053918        0.137050        0.130308        -0.017166       -0.045565       -0.166605       0.066214        0.011122        -0.015967       0.114189        0.014916    0.027770        0.084754        0.176174        -0.083049       0.106313        0.035110        -0.004968       -0.077219       -0.048735       0.009551        -0.031268       0.074965        0.010042        0.047187    0.028760        0.096103        -0.049228       -0.011555       0.032496        0.070770        -0.077987       0.044070        0.018615        -0.018846       0.015677        -0.063793       -0.001495       0.023549    0.038284        -0.004204       -0.055982       -0.045537       -0.033369       -0.046584       0.071702        -0.036919       0.042851        0.013537        0.035471        -0.027192       -0.053215       -0.024522   -0.029496       0.023682        -0.020740       0.026490        -0.025587       -0.016844       -0.035614       0.019064        0.002671        -0.038996       -0.018351       0.019004        -0.019295       0.020606    0.008598        -0.008490
# entity_vec_old's len : 14541

from collections import defaultdict
def newdict(entity_new,entity_old,entity_vec_old):
    dict_old={}
    dict_GAT={}
    for m,n in zip(entity_old,entity_vec_old):
        dict_old[m]=n
    for a,b in zip(entity_old,entity_vec_GAT):
        dict_GAT[a]=b
    with open("data/FB15k-237/multilabel/multilabel_vec_(bat_mlp(2,2)).txt", "w") as f6:
        for e in entity_new:
            print(e)
            if e in dict_old and e in dict_GAT:
                # print(str(dict_old[e]))
                
                # f6.write(str(dict_old[e]))
                # f6.write('	')
                f6.write(str(dict_GAT[e]))
                ro_type=dict_e_types[e]
                print(ro_type)
                for t in types:
                    if t in ro_type:
                        # print(t)
                        f6.write("	1")
                    else:
                        f6.write("	0")
                f6.write('\n')
            # break

newdict(entity_new,entity_old,entity_vec_old)

# ======================================================================================================

# from collections import defaultdict
# def newdict(entity_new,entity_old,entity_vec_old):
#     dict_old={}
#     dict_GAT={}
#     for m,n in zip(entity_old,entity_vec_old):
#         dict_old[m]=n
#     for a,b in zip(entity_old,entity_vec_GAT):
#         dict_GAT[a]=b
#     with open("data/FB15k-237/t-SNE-emb/emb1.txt", "w") as f6:
#         for e in entity_new:
#             print(e)
#             if e in dict_old and e in dict_GAT:
#                 # print(str(dict_old[e]))
                
#                 # f6.write(str(dict_old[e]))
#                 # f6.write('	')
#                 f6.write(str(dict_GAT[e]))
#                 ro_type=dict_e_types[e]
#                 print(ro_type)
#                 for t in types:
#                     if t in ro_type:
#                         # print(t)
#                         f6.write("	1")
#                     else:
#                         f6.write("	0")
#                 f6.write('\n')
#             # break

# newdict(entity_new,entity_old,entity_vec_old)