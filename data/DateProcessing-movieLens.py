import copy
import json
import string
import sys, random, os
from collections import defaultdict

import numpy as np


def loadfile(filename):
    ''' load a file, return a generator. '''
    fp = open(filename, 'r',encoding='gb18030', errors='ignore')
    for i, line in enumerate(fp):
        yield line.strip('\r\n')
        if i % 100000 == 0:
            print('loading %s(%s)' % (filename, i), file=sys.stderr)
    fp.close()
    print('load %s succ' % filename, file=sys.stderr)


def writetofile(data, dfile):
    with open(dfile, 'w') as f:
        for u, ilist in sorted(data.items()):
            items = [li[0] for li in ilist]
            f.write(str(u))
            for item in items:
                f.write(" "+str(item))
            # for i in ilist:
            f.write('\n')


def generate_dataset(filename, pivot=0.7):
    ''' load rating data and split it to training set and test set '''
    trainset_len = 0
    testset_len = 0
    user_train = {}
    user_valid = {}
    user_test = {}
    User = dict()
    fitter_User = dict()
    usermap = dict()
    DATASET = 'ML-1M'
    if not os.path.isdir('./' + DATASET):
        os.mkdir('./' + DATASET)
    train_file = './' + DATASET + '/ML-1M.txt'
    valid_file = './' + DATASET + '/valid.txt'
    test_file = './' + DATASET + '/test.txt'
    train_reverse_file = './' + DATASET + '/train_reverse.txt'
    valid_reverse_file = './' + DATASET + '/valid_reverse.txt'
    test_reverse_file = './' + DATASET + '/test_reverse.txt'
    imap_file = './' + DATASET + '/imap.json'
    umap_file = './' + DATASET + '/umap.json'
    item2attributes_file = './' + DATASET + '_item2attributes.json'

    countU = defaultdict(lambda: 0)  # dict which default 0
    countP = defaultdict(lambda: 0)
    for line in loadfile(filename):
        user, item, rating, time = line.split('::')
        # print('user : ' + user)
        userid = int(user)
        itemid = int(item)

        countU[userid] += 1
        countP[itemid] += 1
    for line in loadfile(filename):
        user, item, rating, time = line.split('::')
        # print('user : ' + user)
        userid = int(user)
        itemid = int(item)

        if countU[userid] < 5 or countP[itemid] < 5:
            continue
        item = str(itemid)
        user = str(userid)
        if userid not in usermap:
            usermap[userid] = 1
            User[userid] = []
        else:
            User[userid].append([item, rating, time])
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

    User = filter_Kcore(User)
    itemremap = dict()


    itemid = 1
    for _, ilist in User.items():
        for i in ilist:
            if i[0] not in itemremap:
                itemremap[i[0]] = itemid
                itemid += 1
    final_User = dict()
    print(len(itemremap))
    for user in User.keys():
        User[user].sort(key=lambda x: x[2])
    for u, ilist in User.items():
        index = 0
        final_User[u] = []
        for i in ilist:
            new_ilist = [itemremap[i[0]], i[1], i[2]]
            index += 1
            final_User[u].append(new_ilist)
    User = final_User
    filter_user = dict()
    for userid in User:
        if len(User[userid]) > 50:
            User[userid] = User[userid][:50]
        if len(User[userid]) >= 5:
            filter_user[userid] = User[userid]
    writetofile(filter_user, train_file)

    datamap = {}
    tagnum = 1
    tagmap = dict()
    attribute_lens = []
    for line in loadfile(moviesfile):
        itemid, _,tags= line.split('::')
        if itemid not in itemremap:
            continue
        itemid = itemremap[itemid]
        tag = tags.split('|')
        datamap[itemid] = []
        for i in tag:
            if i in tagmap:
              x = tagmap[i]
            else:
                x = tagmap[i] = tagnum
                tagnum += 1
            datamap[itemid].append(x)
        attribute_lens.append(len(datamap[itemid]))
    json_str = json.dumps(datamap)
    print(f"Attributes num:{len(tagmap)}")
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')

    with open(item2attributes_file, 'w') as out:
        out.write(json_str)
    itemmap = dict()
    for _, ilist in filter_user.items():
        for i in ilist:
            if i[0] not in itemmap:
                itemmap[i[0]] = 1
    num_instances = sum([len(ilist) for _, ilist in User.items()])
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
if __name__ == '__main__':
    ratingfile = os.path.join('ratings.dat')
    moviesfile = os.path.join('movies.dat')

    generate_dataset(ratingfile,moviesfile)