import gzip
from collections import defaultdict
from datetime import datetime
import os
import copy
import json
import time

import numpy as np
import tqdm


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


countU = defaultdict(lambda: 0)  # dict which default 0
countP = defaultdict(lambda: 0)
line = 0

DATASET = 'Beauty'
dataname = './reviews_Beauty_5.json.gz'
#dataname = '/home/zfan/BDSC/projects/datasets/newamazon_reviews/{}.json.gz'.format(DATASET)
if not os.path.isdir('./'+DATASET):
    os.mkdir('./'+DATASET)
train_file = './'+DATASET+'/Beauty.txt'
valid_file = './'+DATASET+'/valid.txt'
test_file = './'+DATASET+'/test.txt'
imap_file = './'+DATASET+'/imap.json'
umap_file = './'+DATASET+'/umap.json'
item2attributes_file = './'+DATASET+'_item2attributes.json'
train_reverse_file = './'+DATASET+'/train_reverse.txt'
valid_reverse_file = './'+DATASET+'/valid_reverse.txt'
test_reverse_file = './'+DATASET+'/test_reverse.txt'




# for l in parse(dataname):  # get item and user quantity
#     line += 1
#     asin = l['asin']
#     rev = l['reviewerID']
#     time = l['unixReviewTime']
#     countU[rev] += 1
#     countP[asin] += 1

usermap = dict()
usernum = 1
itemmap = dict()
itemnum = 1
User = dict()

for l in parse(dataname):
    line += 1
    asin = l['asin']
    rev = l['reviewerID']
    time = l['unixReviewTime']
    overall = l['overall']
    reviewText = "\""+l['reviewText']+"\""
    reviewTime = "\""+l['reviewTime']+"\""


    #if countU[rev] < 5 or countP[asin] < 5:
    #    continue

    if rev in usermap:  # a user who prior appear
        userid = usermap[rev]
    else:   # new user
        userid = usernum  # assign id
        usermap[rev] = userid # save to map
        User[userid] = []  # construct user list
        usernum += 1
    if asin in itemmap: # ditto
        itemid = itemmap[asin]
    else:
        itemid = itemnum
        itemmap[asin] = itemid
        itemnum += 1
    User[userid].append([itemid, time,reviewTime])  #user list : [(item_id,time),,,,,,]
# sort reviews in User according to time
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item[0]] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # 已经保证Kcore

# 循环过滤 K-core
def filter_Kcore(user_items, user_core=5, item_core=5): # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item[0]] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items
User = filter_Kcore(User)
itemremap = dict()

itemid = 1
for _, ilist in User.items():
    for i in ilist:
        if i[0] not in itemremap:
            itemremap[i[0]] = itemid
            itemid += 1
final_User = dict()
for u, ilist in User.items():
    index = 0
    final_User[u] = []
    for i in ilist:
        new_ilist = [itemremap[i[0]], i[1], i[2]]
        index += 1
        final_User[u].append(new_ilist)
User = final_User

with open(imap_file, 'w') as f:
    json.dump(itemmap, f)

with open(umap_file, 'w') as f:
    json.dump(usermap, f)

filter_user = dict()
for userid in User.keys():
    User[userid].sort(key=lambda x: x[1])
    nfeedback = len(User[userid])
    if nfeedback > 50:
        User[userid] = User[userid][:50]
    if nfeedback < 5:    # if user interaction quantities < 5 , only as train data
        continue
    if len(User[userid]) >=5:
        filter_user[userid]=User[userid]

user_train = {}
user_valid = {}
user_test = {}
user_total = {}

def Amazon_meta(dataset_name,data_maps):
    '''
    asin - ID of the product, e.g. 0000031852
    --"asin": "0000031852",
    title - name of the product
    --"title": "Girls Ballet Tutu Zebra Hot Pink",
    description
    price - price in US dollars (at time of crawl)
    --"price": 3.17,
    imUrl - url of the product image (str)
    --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
    related - related products (also bought, also viewed, bought together, buy after viewing)
    --"related":{
        "also_bought": ["B00JHONN1S"],
        "also_viewed": ["B002BZX8Z6"],
        "bought_together": ["B002BZX8Z6"]
    },
    salesRank - sales rank information
    --"salesRank": {"Toys & Games": 211836}
    brand - brand name
    --"brand": "Coxlures",
    categories - list of categories the product belongs to
    --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    '''
    datas = {}
    meta_flie = './meta_' + dataset_name + '.json.gz'
    item_asins = list(data_maps.keys())
    for info in parse(meta_flie):
        if info['asin'] not in item_asins:
            continue
        datas[info['asin']] = info
    return datas

metamap = Amazon_meta(DATASET,itemmap)
print(usernum, itemnum)

# categories 和 brand is all attribute
def get_attribute_Amazon(meta_infos, datamaps):
    attribute_core = 0
    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        for cates in info['categories']:
            for cate in cates[1:]: # 把主类删除 没有用
                attributes[cate] +=1
        try:
            attributes[info['brand']] += 1
        except:
            pass

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        try:
            if attributes[info['brand']] >= attribute_core:
                new_meta[iid].append(info['brand'])
        except:
            pass
        for cates in info['categories']:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = itemremap[datamaps[iid]] # start at 1
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'after delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # # 更新datamap
    # datamaps['attribute2id'] = attribute2id
    # datamaps['id2attribute'] = id2attribute
    # datamaps['attributeid2num'] = attributeid2num
    json_str = json.dumps(items2attributes)

    with open(item2attributes_file, 'w') as out:
        out.write(json_str)
    return datamaps, items2attributes

get_attribute_Amazon(metamap,itemmap)

def writetofile(data, dfile):
    with open(dfile, 'w') as f:
        for u, ilist in sorted(data.items()):
            items = [li[0] for li in ilist]
            f.write(str(u))
            for item in items:
                f.write(" "+str(item))
            # for i in ilist:
            f.write('\n')

    # with open(dfile, 'w') as f:
    #     for u, ilist in sorted(data.items()):
    #         for i, t, reviewText, reviewTime,rating in ilist:
    #             f.write(str(u) + '\t'+ str(i) + '\t' + str(t) + "\n")
                # f.write(str(u) + '\t'+ str(i) + '\t' + str(t) + '\t'+str(rating)+ '\t' + str(reviewText) + '\t' + str(reviewTime) +"\n")

writetofile(filter_user, train_file)


# writetofile(user_train_reverse, train_reverse_file)
# writetofile(user_valid_reverse, valid_reverse_file)
# writetofile(user_test_reverse, test_reverse_file)

num_instances = sum([len(ilist) for _, ilist in User.items()])
print('total user: ', len(User))
print('total instances: ', num_instances)
print('total items: ', itemnum)
print('density: ', num_instances / (len(User) * itemnum))
print('valid #users: ', len(user_valid))
numvalid_instances = sum([len(ilist) for _, ilist in user_valid.items()])
print('valid instances: ', numvalid_instances)
numtest_instances = sum([len(ilist) for _, ilist in user_test.items()])
print('test #users: ', len(user_test))
print('test instances: ', numtest_instances)
