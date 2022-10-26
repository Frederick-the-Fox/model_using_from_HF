# coding=utf-8
# Copyright (c) 2022, Frederick Yuanchun Wang. wyc99@mail.nwpu.edu.cn 
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#             佛祖保佑   🙏   永无BUG 
#
#                    _ooOoo_
#                   o8888888o
#                   88" . "88
#                   (| -_- |)
#                   O\  =  /O
#                ____/`---'\____
#              .'  \\|     |//  `.
#             /  \\|||  :  |||//  \
#            /  _||||| -:- |||||-  \
#            |   | \\\  -  /// |   |
#            | \_|  ''\---/''  |   |
#            \  .-\__  `-`  ___/-. /
#          ___`. .'  /--.--\  `. . __
#       ."" '<  `.___\_<|>_/___.'  >'"".
#      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#      \  \ `-.   \_ __\ /__ _/   .-` /  /
# ======`-.____`-.___\_____/___.-`____.-'======
#                    `=---='
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# happy 1024 hh

from asyncio import AbstractEventLoop
from re import L
from symbol import testlist_comp
import torch
from torch import nn
from torch import optim
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch.utils.data as Data
import numpy as np
from loguru import logger
import argparse 
import json
import xlrd
import os
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser() # 初始化

parser.add_argument('--data_dir', default='/data/wangyuanchun/NLP_course/dataset/post_processed',
                       help='data_root')
parser.add_argument('--batch_size', type=int, default=1,
                       help='size of one batch') 
parser.add_argument('--epoch', type=int, default=100,
                       help='epoch')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='learning_rate')
parser.add_argument('--val_per_ite', type=int, default=100,
                       help='validation per how many iterations')
parser.add_argument('--model_save', default='/data/wangyuanchun/NLP_course/saved_models/',
                       help='validation per how many iterations')
parser.add_argument('--weight', type = int, default=5000,
                       help='validation per how many iterations')
parser.add_argument('--device', default='cuda:0',
                       help='the device you want to use to train')
parser.add_argument('--local_rank', type = int, default=-1)
args, others_list = parser.parse_known_args() # 解析已知参数

# DDP参数初始化
local_rank = args.local_rank
torch.cuda.set_device(args.local_rank)

# DDP后端初始化
device = torch.device("cuda", local_rank)
torch.distributed.init_process_group(backend='nccl')

# print(args)
# print(args.b)
# parser.add_argument('--c')

# others = parser.parse_args(others_list)
# print(others)
# print(others.c)

def load_data(arg_mode):
    # """用来生成训练、测试数据"""
    # train_df = pd.read_csv("bert_example.csv", header=None)
    # sentences = train_df[0].values
    # targets = train_df[1].values
    # train_inputs, test_inputs, train_targets, test_targets = train_test_split(sentences, targets)
    if arg_mode == 'train':
        data_dir = '/data/wangyuanchun/NLP_course/dataset/post_processed/train_mse_2_line.json'
        data_dir = '/data/wangyuanchun/NLP_course/codes/train_example.json'
    elif arg_mode == 'val':
        data_dir = '/data/wangyuanchun/NLP_course/dataset/post_processed/val_mse_2_line.json'
        data_dir = '/data/wangyuanchun/NLP_course/codes/val_example.json'

    train_inputs = []
    train_targets = []
    with open(data_dir, 'r') as file_src:
        train_src = json.load(file_src)
    file_src.close()

    for each_train in train_src:
        train_inputs.append(each_train['text'])
        train_targets.append(torch.tensor(each_train['label'], dtype=torch.float))
    # with open(data_dir + 'dev.json', 'r') as file_src:
    #     val_src = json.load(file_src)
    # file_src.close()
    
    # train_inputs = train_src['text']
    # val_inputs = val_src['text']
    # train_targets = train_src['label']
    # val_targets = val_src['label']
    return train_inputs, train_targets

class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese")

        self.use_bert_classify = nn.Linear(768, 40474) 
        self.class_transfer = nn.Linear(1 * args.batch_size, 2) 
        self.sig_mod = nn.Sigmoid()
        self.sm = torch.nn.Softmax(dim = -1)

    def forward(self, batch_sentences):
        sentence_tokenized = self.tokenizer(batch_sentences,
                                            truncation=True,
                                            padding=True,  
                                            max_length=30,  
                                            add_special_tokens=True)  
        input_ids = torch.tensor(sentence_tokenized['input_ids']).to(device) 
        attention_mask = torch.tensor(sentence_tokenized['attention_mask']).to(device) 
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :] 
        # logits = self.use_bert_classify(bert_cls_hidden_state)
        logits = bert_cls_hidden_state
        logits_t = torch.t(logits)
        class_t = self.class_transfer(logits_t)
        class_ori = torch.t(class_t)
        class_pre = self.use_bert_classify(class_ori)
        output = self.sm(torch.t(class_pre))

        # return self.sig_mod(linear_output)
        # output = self.sig_mod(logits)
        return output


def train(args):
    print("start loading data...")
    train_inputs, train_targets = load_data('train')

    print("data loaded! ")
    # epochs = args.epoch
    # batch_size = args.batch_size
    # data_dir = args.data_dir
    epochs = args.epoch
    batch_size = args.batch_size

    # DDP利用随机种子采样的封装好的方法
    train_sampler = torch.utils.data.distributed.DistributedSampler

    train_sentence_loader = Data.DataLoader(
        dataset=train_inputs,
        batch_size=batch_size,  # 每块的大小
    )
    train_label_loader = Data.DataLoader(
        dataset=train_targets,
        batch_size=batch_size,
    )
    
    bert_classifier_model = BertClassificationModel()
    model = bert_classifier_model.to(local_rank)
    #DDP 包装model
    if torch.distributed.get_rank() == 0 and os.path.exists(args.model_save + '/saved.pkl'):
        model.load_state_dict(torch.load(args.model_save))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    #用DDP包装后的model的参数来初始化迭代器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # criterion = nn.MultiLabelSoftMarginLoss()
    criterion = torch.nn.MSELoss(reduction='mean')
    criterion = criterion.to(local_rank)

    print("training start...")
    bert_classifier_model.train()

    iteration = 0

    for epoch in range(epochs): # 开始训练
        print('...this is epoch : {}...'.format(epoch))
        loss_list = []       
        for sentences, labels in zip(train_sentence_loader, train_label_loader):
            sentences = sentences
            labels = labels.to(local_rank)
            # print (labels)
            # break
            optimizer.zero_grad()
            result = bert_classifier_model(sentences)
            # print(result)
            # print(labels)
            loss = torch.tensor(0)
            loss_ite = 0
            best_eval = 0
            es_count = 0
            for each_result in result:
                # print('labels:{}'.format(labels[0][loss_ite]))
                # print('each_result:{}'.format(each_result))
                # print('labels[loss_ite]:{}'.format(labels[loss_ite]))
                # print('labels[loss_ite].detach().numpy():{}'.format(labels[loss_ite].detach().numpy()[0][0]))
                if labels[0].cpu().detach().numpy()[loss_ite][0] == 1:
                    loss = loss + args.weight * criterion(each_result, labels[0][loss_ite])
                else:
                    loss = loss + criterion(each_result, labels[0][loss_ite])
                loss_ite += 1
            
            loss = loss / args.weight
            print('this is iteration : {} and loss : {}'.format(iteration, loss.item()))
            writer.add_scalar("loss",loss,iteration)
            loss.backward()
            optimizer.step()
            # break
            loss_list.append(loss.cpu().detach().numpy())
            iteration += 1
            if iteration and iteration % args.val_per_ite == 0:
                torch.save(bert_classifier_model.module.state_dict(), args.model_save + '/saved.pkl')
                bert_classifier_model.eval()
                acc, sentence_acc = eval(args)
                writer.add_scalar("binary_acc",acc,iteration)
                writer.add_scalar("sentence_acc",sentence_acc,iteration)
                if acc >= best_eval:
                    best_eval = acc
                    es_count = 0
                else:
                    es_count += 1
                
                if es_count == 30:
                    print('early_stopping!!!')
                    break
        
        if torch.distributed.get_rank() == 0:
            torch.save(bert_classifier_model.module.state_dict(), args.model_save + '/final.pkl')


def eval(args):
    print("eval : start loading data...")
    train_inputs, train_targets = load_data('val')

    print("data loaded! ")
    # epochs = args.epoch
    # batch_size = args.batch_size
    # data_dir = args.data_dir
    batch_size = 1

    train_sentence_loader = Data.DataLoader(
        dataset=train_inputs,
        batch_size=batch_size,  # 每块的大小
    )
    train_label_loader = Data.DataLoader(
        dataset=train_targets,
        batch_size=batch_size,
    )
    bert_classifier_model = BertClassificationModel()
    bert_classifier_model.load_state_dict(torch.load(args.model_save + '/saved.pkl'))
    bert_classifier_model.to(device)

    print("model saved in " + args.model_save + '/saved.pkl' + " validating start...")
    bert_classifier_model.eval()

    sentence_cnt = 0
    sentence_correct_cnt = 0
    sentence_inco_cnt = 0
    binary_acc_list = []

    for sentences, labels in zip(train_sentence_loader, train_label_loader):

        labels = labels.to(device)
        result = bert_classifier_model(sentences).cpu()
        loss_ite = 0
        correct_cnt = 0
        inco_cnt = 0
        for each_result in result:

            if labels[0].cpu().detach().numpy()[loss_ite][0] == 1:
                if torch.argmax(each_result).item() == 0:
                    correct_cnt += 1
                else:
                    inco_cnt += 1
            else:
                if torch.argmax(each_result).item() == 1:
                    correct_cnt += 1
                else:
                    inco_cnt += 1
        acc = correct_cnt / (correct_cnt + inco_cnt)
        binary_acc_list.append(acc)
        if acc == 1:
            sentence_correct_cnt += 1
        else:
            sentence_inco_cnt += 1
        sentence_cnt += 1
        if sentence_cnt == 100:
            break

    sentence_acc = sentence_correct_cnt / (sentence_correct_cnt + sentence_inco_cnt)
    binary_acc = np.mean(binary_acc_list)
    print('validation end, binary_acc: {}; sentence_acc:{}'.format(binary_acc, sentence_acc))
    return binary_acc, sentence_acc

def test():
    print("doint test : start loading data...")

    excel_path = "/data/wangyuanchun/NLP_course/dataset/国际疾病分类 ICD-10北京临床版v601.xlsx"
    excel = xlrd.open_workbook(excel_path,encoding_override="utf-8")
    sheet = excel.sheets()[0]
    sheet_row_mount = sheet.nrows
    sheet_col_mount = sheet.ncols
    print("row number: {0} ; col number: {1}".format(sheet_row_mount, sheet_col_mount))

    item_list = []

    for x in range(0, sheet_row_mount):
        y = 1
        item_list.append(sheet.cell_value(x, y))

    #LOADING MODEL
    bert_classifier_model = BertClassificationModel()
    bert_classifier_model.load_state_dict(torch.load('../saved_models/saved.pkl'))
    bert_classifier_model.eval()

    with open('../dataset/test.json', 'r') as file_src:
        data_src = json.load(file_src)
    file_src.close()

    # test_inputs = []
    count = 0
    data_json_list = []
    for each_test in data_src:
        # test_inputs.append([each_test['text']])
        sentence = [each_test['text']]
    
    # test_sentence_loader = Data.DataLoader(
    #     dataset=test_inputs,
    #     batch_size=1,  # 每块的大小
    # )

    # for sentence in zip(test_inputs):
        # for each_json in data_src:
        print('this is sentence {}'.format(count))
        count += 1
        
        result = bert_classifier_model(sentence)
        target_json = {}
        target_json["text"] = sentence[0]
        result_list = []
        item_ite = 0
        result_max = max(result[0].detach().numpy())
        result_mean = np.mean(result[0].detach().numpy())
        th = result_max - (result_max - result_mean) / 2
        # print(result)
        print(np.mean(result[0].detach().numpy()))
        for each_value in result[0]:
            if each_value.item() > th:
                result_list.append(item_list[item_ite])
            item_ite += 1
        
        target_json["normalized_result"] = str.join('##', result_list)
        data_json_list.append(target_json)
        if count == 10:
            break

    js_str = json.dumps(data_json_list, indent=4, ensure_ascii=False)

    with open('./final_result.json', 'w') as output:
        output.write(js_str)
    output.close()

if __name__ == '__main__':
    time = datetime.datetime.now()
    time_str = str(time.month) + '-' + str(time.day) + '-' + str(time.hour) + '-' + str(time.minute)
    writer = SummaryWriter('./runs/' + time_str)
    args.model_save = args.model_save + time_str
    os.system("mkdir " + args.model_save)
    train(args)
    # eval(args)
    # test()
    print('done :-)')