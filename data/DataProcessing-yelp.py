import gzip
import io
import sys
from collections import defaultdict
from datetime import datetime
import os
import copy
import json
import time
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')         #改变标准输出的默认编码
# def parse(path):
#     with open(path=path,"r") as f:
#         g =
#     g = json.loads(path)
#     for l in g:
#         yield eval(l)
import numpy as np
import tqdm

countU = defaultdict(lambda: 0)  # dict which default 0
countP = defaultdict(lambda: 0)
line = 0

DATASET = 'yelp'
dataname = './yelp_academic_dataset_review.json'
#dataname = '/home/zfan/BDSC/projects/datasets/newamazon_reviews/{}.json.gz'.format(DATASET)
if not os.path.isdir('./'+DATASET):
    os.mkdir('./'+DATASET)
train_file = './'+DATASET+'/Yelp.txt'
valid_file = './'+DATASET+'/valid.txt'
test_file = './'+DATASET+'/test.txt'
imap_file = './'+DATASET+'/imap.json'
umap_file = './'+DATASET+'/umap.json'

train_reverse_file = './'+DATASET+'/train_reverse.txt'
valid_reverse_file = './'+DATASET+'/valid_reverse.txt'
test_reverse_file = './'+DATASET+'/test_reverse.txt'
item2attributes_file = './'+DATASET+'_item2attributes.json'

usermap = dict()
usernum = 1
itemmap = dict()
itemnum = 1
User = dict()
date_max = '2019-12-31 00:00:00'
date_min = '2019-01-01 00:00:00'
data_flie = './yelp_academic_dataset_review.json'
lines = open(data_flie,encoding="utf8").readlines()
for line in tqdm.tqdm(lines):
    l = json.loads(line.strip())
    asin = l['business_id']
    rev = l['user_id']
    # time = l['date']
    date = l['date']
    score = l['stars']
    text = l['text']
    if date < date_min or date > date_max or float(score) <= 0.0:
        continue
    countU[rev] += 1
    countP[asin] += 1


for line in tqdm.tqdm(lines):
    l = json.loads(line.strip())
    # line += 1
    asin = l['business_id']
    rev = l['user_id']
    # time = l['date']
    date = l['date']
    score = l['stars']
    text = l['text']
    if date < date_min or date > date_max:
        continue

    if countU[rev] < 5 or countP[asin] < 5:
       continue
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timestamp = int(time.mktime(timeArray))
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
    User[userid].append([itemid,text, timestamp,date,score])  #user list : [(item_id,time),,,,,,]
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
        return user_count, item_count, True  # 已经保证Kcore

    # 循环过滤 K-core


def filter_Kcore(user_items, user_core=5, item_core=5):  # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:  # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item[0]] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items
with open(imap_file, 'w') as f:
    json.dump(itemmap, f)

with open(umap_file, 'w') as f:
    json.dump(usermap, f)

for userid in User.keys():
    User[userid].sort(key=lambda x: x[2])
User = filter_Kcore(User)
def Yelp_meta(datamaps):
    meta_infos = {}
    meta_file = './yelp_academic_dataset_business.json'
    item_ids = list(datamaps.keys())
    lines = open(meta_file,encoding="utf8").readlines()
    for line in tqdm.tqdm(lines):
        info = json.loads(line)
        if info['business_id'] not in item_ids:
            continue
        meta_infos[info['business_id']] = info
    return meta_infos

def get_attribute_Yelp(meta_infos, datamaps, attribute_core=0):
    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        try:
            cates = [cate.strip() for cate in info['categories'].split(',')]
            for cate in cates:
                attributes[cate] +=1
        except:
            pass
    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []
        try:
            cates = [cate.strip() for cate in info['categories'].split(',') ]
            for cate in cates:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
        except:
            pass
    # 做映射
    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in new_meta.items():
        item_id = datamaps[iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'after delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    # 更新datamap
    # datamaps['attribute2id'] = attribute2id
    # datamaps['id2attribute'] = id2attribute
    json_str = json.dumps(items2attributes)

    with open(item2attributes_file, 'w') as out:
        out.write(json_str)
    return datamaps, items2attributes
    # return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes

meta_infos = Yelp_meta(itemmap)
get_attribute_Yelp(meta_infos,itemmap)
for userid in User.keys():
    User[userid].sort(key=lambda x: x[1])
user_train = {}
user_valid = {}
user_test = {}
new_user = 1
filter_user = dict()
for user in User:
    nfeedback = len(User[user])
    if nfeedback > 50:
        User[user] = User[user][:50]
    if nfeedback < 5:  # if user interaction quantities < 3 , only as train data
        continue
    else:
        filter_user[new_user] = User[user]
        new_user += 1

def writetofile(data, dfile):
    with open(dfile, 'w') as f:
        for u, ilist in sorted(data.items()):
            items = [li[0] for li in ilist]
            f.write(str(u))
            for item in items:
                f.write(" "+str(item))
            f.write('\n')

writetofile(filter_user, train_file)

itemmap = dict()
for _, ilist in filter_user.items():
    for i in ilist:
        if i[0] not in itemmap:
            itemmap[i[0]] = 1
        else:
            itemmap[i[0]] += 1
item_fitter = 0
for item in itemmap:
    if itemmap[item]<5:
        item_fitter+=1
print("item_fitter",item_fitter)
maxlen = 0
minlen = 1000000
avglen = 0
for _, ilist in filter_user.items():
    listlen = len(ilist)
    maxlen = max(maxlen, listlen)
    minlen = min(minlen, listlen)
    avglen += listlen
avglen /= len(filter_user)
print('max length: ', maxlen)
print('min length: ', minlen)
print('avg length: ', avglen)
num_instances = sum([len(ilist) for _, ilist in filter_user.items()])
print('total user: ', len(filter_user))
print('total instances: ', num_instances)
print('total items: ', len(itemmap))
print('density: ', num_instances / (len(filter_user) * len(itemmap)))
print('valid #users: ', len(user_valid))
numvalid_instances = sum([len(ilist) for _, ilist in user_valid.items()])
print('valid instances: ', numvalid_instances)
numtest_instances = sum([len(ilist) for _, ilist in user_test.items()])
print('test #users: ', len(user_test))
print('test instances: ', numtest_instances)
