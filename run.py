# coding: UTF-8
import time
import torch
import argparse
import logging as lgg
import transformers.utils.logging
from train_mutual_learning import train
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif
from transformers import RobertaTokenizer, BertTokenizer, AlbertTokenizer, DistilBertTokenizer, GPT2Tokenizer
from datetime import datetime
import warnings
import numpy as np
warnings.filterwarnings("ignore")
transformers.utils.logging.set_verbosity_error()

########## CHOOSE MODEL ##########
#model = 'roberta-base'
#model = 'bert-base-uncased'
#model = 'albert-base-v2'
model = 'distilbert-base-uncased'
#model = 'roberta-large'
##################################


def setlogging(level, filename):
    for handler in lgg.root.handlers[:]:
        lgg.root.removeHandler(handler)
    lgg.basicConfig(level=level,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%H:%M',
                    filename=filename,
                    filemode='w')
    logc = lgg.StreamHandler()
    logc.setLevel(level=lgg.DEBUG)
    logc.setFormatter(lgg.Formatter('%(message)s'))
    lgg.getLogger().addHandler(logc)


class Config(object):
    def __init__(self, dataset, cuda=0, finetune=True, lambda4=1.0):
        self.model_name = model
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'

        self.i2conn = [x.strip() for x in open(dataset + '/data/conn.txt').readlines()]
        self.conn2i = dict((x, xid) for xid, x in enumerate(self.i2conn))
        self.i2top = [x.strip() for x in open(dataset + '/data/class4.txt').readlines()]
        self.top2i = dict((x, xid) for xid, x in enumerate(self.i2top))
        self.i2sec = [x.strip() for x in open(dataset + '/data/class11.txt').readlines()]
        self.sec2i = dict((x, xid) for xid, x in enumerate(self.i2sec))

        self.save_path_top = dataset + '/saved_dict/' + self.model_name + '_top.ckpt'        # 
        self.save_path_sec = dataset + '/saved_dict/' + self.model_name + '_sec.ckpt'        # 
        self.save_path_conn = dataset + '/saved_dict/' + self.model_name + '_conn.ckpt'
        t = datetime.now().strftime('%B%d-%H:%M:%S')
        self.log = dataset + '/log/' + self.model_name + '.log'
        print("LOG PATH:",self.log)
        self.device = torch.device('cuda:{0}'.format(cuda) if torch.cuda.is_available() else 'cpu')   # 

        self.require_improvement = 10000
        self.n_top = len(self.i2top)
        self.n_sec = len(self.i2sec)
        self.n_conn = len(self.i2conn)
        self.pad_size = 100
        self.finetune_bert = finetune
        if self.finetune_bert:
            self.num_epochs = 15
            self.learning_rate = 10e-5
        else:
            self.num_epochs = 20
            self.learning_rate = 0.00005

        if model == 'bert-base-uncased':
            self.batch_size = 32
            self.hidden_size = 768
            self.bert_path = model
            self.tokenizer = BertTokenizer.from_pretrained(model)
        elif model == 'albert-base-v2':
            self.batch_size = 32
            self.hidden_size = 768
            self.bert_path = model
            self.tokenizer = AlbertTokenizer.from_pretrained(model)
        elif model == 'distilbert-base-uncased':
            self.batch_size = 32
            self.hidden_size = 768
            self.bert_path = model
            self.tokenizer = DistilBertTokenizer.from_pretrained(model)
        elif model == 'roberta-base':
            self.batch_size = 32
            self.hidden_size = 768
            self.bert_path = model
            self.tokenizer = RobertaTokenizer.from_pretrained(model)
        elif model == 'roberta-large':
            self.batch_size = 16
            self.hidden_size = 1024
            self.bert_path = model
            self.tokenizer = RobertaTokenizer.from_pretrained(model)

        

        self.lambda1 = 1.0    # for the top-level loss
        self.lambda2 = 1.0    # for the second-level loss
        self.lambda3 = 1.0    # for the conn-level loss
        self.lambda4 = lambda4
        self.x_dim = self.hidden_size

        self.need_clc_loss = True
        # for conn results
        self.need_conn_rst = True

        # model
        self.model_choose = "bert_simple"

        # show training and test time
        self.show_time = True

        self.num_gcn_layer = 6  # gcn layer num
        self.label_num = 117    # total label num(top:4,second:11,conn:102)
        self.label_embedding_size = 100
        if model == 'roberta-large':
            self.attn_hidden_size = 1024
            self.enc_emb_dim = 512
            self.dec_emb_dim = 512
            self.enc_hidden_size = 1024
            self.dec_hidden_size = 1024
            self.enc_hid_dim = 1024
            self.dec_hid_dim = 1024
        else:
            self.attn_hidden_size = 768
            self.enc_emb_dim = 384
            self.dec_emb_dim = 384
            self.enc_hidden_size = 768
            self.dec_hidden_size = 768
            self.enc_hid_dim = 768
            self.dec_hid_dim = 768


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label Dependence-aware Sequence Generation Model for Multi-level Implicit Discourse Relation Recognition')
    parser.add_argument('--model', type=str, default=model, help='choose a model')
    parser.add_argument('--cuda', type=int, default=0, choices=[0, 1], help='choose a cuda: 0 or 1')
    parser.add_argument('--tune', type=int, default=1, choices=[1, 0], help='fine tune or not: 0 or 1')
    parser.add_argument('--base', type=str, default='roberta', choices=['roberta'], help='roberta model as encoder')
    parser.add_argument('--lambda4', type=float, default=1.0, help='lambda for kl loss')
    args = parser.parse_args()

    dataset = 'PDTB/Ji'  
    model_name = args.model      # roberta-base / bert-base-uncased
    x = import_module(model_name)
    config = Config(dataset, args.cuda, bool(args.tune), args.lambda4)
    setlogging(lgg.DEBUG, config.log)

    hyper_parameters = config.__dict__.copy()
    hyper_parameters['i2conn'] = ''
    hyper_parameters['conn2i'] = ''
    hyper_parameters['i2top'] = ''
    hyper_parameters['top2i'] = ''
    hyper_parameters['i2sec'] = ''
    hyper_parameters['sec2i'] = ''
    hyper_parameters['tokenizer'] = ''
    lgg.info(hyper_parameters)
    start_time = time.time()
    lgg.info("Loading data...")

    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    lgg.info("Time usage: {}".format(time_dif))

    # train
    bertmodel = x.Model(config).to(config.device)
    train(config, bertmodel, train_iter, dev_iter, test_iter)
