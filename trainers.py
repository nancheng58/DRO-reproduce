# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 11:06
# @Author  : Hui Wang

import numpy as np
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = False and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    # length: [length_lower_bound, length_upper_bound)
    def get_sample_scores_length(self, epoch, answers, pred_list, original_input_length, length_lower_bound, length_upper_bound):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        filter_pred_list = []
        for i in range(len(original_input_length)):  # length filter
            if length_lower_bound <= original_input_length[i] and original_input_length[i] < length_upper_bound:
                filter_pred_list.append(pred_list[i])
        pred_list = np.array(filter_pred_list)
        R_5, NDCG_5, MRR_5 = get_metric(pred_list, 5)
        R_10, NDCG_10, MRR_10 = get_metric(pred_list, 10)
        R_20, NDCG_20, MRR_20 = get_metric(pred_list, 20)

        post_fix = {
            "Epoch": epoch,
            "HR_5": '{:.7f}'.format(R_5), "HR_10": '{:.7f}'.format(R_10), "HR_20": '{:.7f}'.format(R_20),
            "NDCG@5": '{:.7f}'.format(NDCG_5), "NDCG@10": '{:.7f}'.format(NDCG_10), "NDCG@20": '{:.7f}'.format(NDCG_20),
            "MRR@5": '{:.7f}'.format(MRR_5), "MRR@10": '{:.7f}'.format(MRR_10), "MRR@20": '{:.7f}'.format(MRR_20)
        }
        print(str(length_lower_bound) + " " + str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(length_lower_bound) + " " + str(post_fix) + '\n')
        return str(post_fix)

    def get_sample_scores(self, epoch, answers, pred_list, original_input_length):
        length_lower_bound = [0, 20, 30, 40]
        length_upper_bound = [20, 30, 40, 51]
        for i in range(len(length_lower_bound)):
            self.get_sample_scores_length(epoch, answers, pred_list, original_input_length, length_lower_bound[i], length_upper_bound[i])
        # print(post_fix)
        # with open(self.args.log_file, 'a') as f:
        #     f.write(str(post_fix) + '\n')
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        # HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        # R_20 = recall_at_k(answers, pred_list, 20)
        # R_50 = recall_at_k(answers, pred_list, 50)
        R_5, NDCG_5, MRR_5 = get_metric(pred_list, 5)
        R_10, NDCG_10, MRR_10 = get_metric(pred_list, 10)
        R_20, NDCG_20, MRR_20 = get_metric(pred_list, 20)

        post_fix = {
            "Epoch": epoch,
            "HR_5": '{:.7f}'.format(R_5), "HR_10": '{:.7f}'.format(R_10), "HR_20": '{:.7f}'.format(R_20),
            "NDCG@5": '{:.7f}'.format(NDCG_5), "NDCG@10": '{:.7f}'.format(NDCG_10), "NDCG@20": '{:.7f}'.format(NDCG_20),
            "MRR@5": '{:.7f}'.format(MRR_5), "MRR@10": '{:.7f}'.format(MRR_10), "MRR@20": '{:.7f}'.format(MRR_20)
        }
        return [R_5, R_10, R_20, NDCG_5, NDCG_10, NDCG_20, MRR_5, MRR_10, MRR_20], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def distributionally_robust_optimization(self, seq_out):  # DRO
        item_matrix = self.model.item_embeddings.weight
        dro_output = torch.matmul(seq_out, item_matrix.transpose(0, 1))
        dro_output = self.model.dro_act(dro_output)
        rating_pred = torch.logsumexp(dro_output, dim=1).mean()
        return rating_pred

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class SASRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(SASRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            avg_loss = 0.0
            rec_avg_loss = 0.0
            dro_avg_loss = 0.0
            a = 0.1
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _, _ = batch
                # Binary cross_entropy
                sequence_output = self.model.sasrec(input_ids)
                recommend_output = sequence_output[:, -1, :]
                # 推荐的结果
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)
                dro_loss = self.distributionally_robust_optimization(recommend_output)

                loss = rec_loss + a * dro_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                avg_loss += loss.item()
                rec_avg_loss += rec_loss.item()
                dro_avg_loss += dro_loss.item()

            post_fix = {
                "epoch": epoch,
                "loss": '{:.4f}'.format(avg_loss / len(rec_data_iter)),
                "rec_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "dro_loss": '{:.4f}'.format(dro_avg_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, _ = batch
                    recommend_output = self.model.sasrec(input_ids)

                    recommend_output = recommend_output[:, -1, :]
                    # 推荐的结果

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                answer_list = None
                original_input_length = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs, original_input_length_batch = batch
                    recommend_output = self.model.sasrec(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                        answer_list = answers.cpu().data.numpy()
                        original_input_length = original_input_length_batch.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        original_input_length = np.append(original_input_length, original_input_length_batch.cpu().data.numpy(), axis=0)

                return self.get_sample_scores(epoch, answer_list, pred_list, original_input_length)
