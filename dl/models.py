import argparse
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import pandas as pd

from dl.reparameterization import apply_weight_norm, remove_weight_norm
from dl.model import SentimentClassifier
from dl.configure_data import DataConfig

class Args:
    def __init__(self):
        self.all_layers = False
        self.alphabet_size = 128
        self.batch_size = 128
        self.chkpt_grad = False
        self.classifier_dropout = 0.1
        self.classifier_hidden_activation = 'PReLU'
        self.classifier_hidden_layers = None
        self.clip = 0
        self.concat_max = False
        self.concat_mean = False
        self.concat_min = False
        self.constant_decay = None
        self.cuda = False
        self.data = ['data.csv']
        self.data_set_type = 'supervised'
        self.data_size = 257
        self.delim = ','
        self.distributed_backend = 'gloo'
        self.double_thresh = False
        self.dropout = 0.0
        self.dual_thresh = False
        self.dynamic_loss_scale = False
        self.emsize = 64
        self.epochs = 5
        self.eval_batch_size = 0
        self.eval_label_key = None
        self.eval_seq_length = 256
        self.eval_text_key = None
        self.fp16 = False
        self.get_hidden = False
        self.heads_per_class = 1
        self.joint_binary_train = False
        self.label_key = 'label'
        self.lazy = False
        self.load = 'dl/mlstm_semeval.clf'
        self.load_optim = False
        self.log_interval = 100
        self.loose_json = False
        self.loss_scale = 1
        self.lr = 0.0005
        self.max_seq_len = None
        self.model = 'mLSTM'
        self.multinode_init = False
        self.ncontext = 2
        self.neural_alphabet = False
        self.neurons = 1
        self.nhid = 4096
        self.nlayers = 1
        self.no_weight_norm = False
        self.non_binary_cols = None
        self.num_hidden_warmup = 0
        self.optim = 'Adam'
        self.padding_idx = 0
        self.persist_state = 0
        self.preprocess = False
        self.process_fn = 'process_str'
        self.rank = -1
        self.residuals = False
        self.samples_per_shard = 100
        self.save = 'lang_model.pt'
        self.save_iters = 10000
        self.save_optim = False
        self.save_probs = 'clf_results.npy'
        self.seed = 1234
        self.seq_length = 256
        self.shuffle = False
        self.split = '1.'
        self.test = None
        self.text_key = 'text'
        self.tied = False
        self.tokenizer_model_type = 'bpe'
        self.tokenizer_path = 'tokenizer.model'
        self.tokenizer_type = 'CharacterLevelTokenizer'
        self.transpose = False
        self.use_softmax = False
        self.valid = None
        self.vocab_size = 256
        self.world_size = 1
        self.write_results = ''


class SentimentDiscovery:
    def get_data(self):
        data_config = DataConfig(parser=None, defaults={
                'world_size': 1,
                'rank': -1,
                'persist_state': 0,
                'lazy': False,
                'shuffle': False,
                'transpose': False,
                'data_set_type': 'supervised',
                'seq_length': 256,
                'eval_seq_length': 256,
                'samples_per_shard': 100
               })

        self.args.cuda = torch.cuda.is_available()
        self.args.shuffle = False

        if self.args.seed is not -1:
            torch.manual_seed(self.args.seed)
            if self.args.cuda:
                torch.cuda.manual_seed(self.args.seed)

        (train_data, val_data, test_data), tokenizer = data_config.apply(self.args)
        self.args.data_size = tokenizer.num_tokens
        self.args.padding_idx = tokenizer.command_name_map['pad'].Id
        return (train_data, val_data, test_data), tokenizer

    def get_model(self):
        sd = None
        self.args.cuda = torch.cuda.is_available()
        self.args.shuffle = False
        model_args = self.args
        if self.args.load is not None and self.args.load != '':
            if not self.args.cuda:
                sd = torch.load(self.args.load, map_location=torch.device('cpu'))
            else:
                sd = torch.load(self.args.load)
            if 'args' in sd:
                model_args = sd['args']
            if 'sd' in sd:
                sd = sd['sd']

        ntokens = model_args.data_size
        concat_pools = model_args.concat_max, model_args.concat_min, model_args.concat_mean
        if self.args.model == 'transformer':
            model = SentimentClassifier(model_args.model, ntokens, None, None, None,
                                        model_args.classifier_hidden_layers, model_args.classifier_dropout,
                                        None, concat_pools, False, model_args)
        else:
            model = SentimentClassifier(model_args.model, ntokens, model_args.emsize, model_args.nhid,
                                        model_args.nlayers,
                                        model_args.classifier_hidden_layers, model_args.classifier_dropout,
                                        model_args.all_layers, concat_pools, False, model_args)
        self.args.heads_per_class = model_args.heads_per_class
        self.args.use_softmax = model_args.use_softmax
        try:
            self.args.classes = list(model_args.classes)
        except:
            self.args.classes = [self.args.label_key]

        try:
            self.args.dual_thresh = model_args.dual_thresh and not model_args.joint_binary_train
        except:
            self.args.dual_thresh = False

        if self.args.cuda:
            model.cuda()

        if self.args.fp16:
            model.half()

        if sd is not None:
            try:
                model.load_state_dict(sd)
            except:
                # if state dict has weight normalized parameters apply and remove weight norm to model while loading sd
                if hasattr(model.lm_encoder, 'rnn'):
                    apply_weight_norm(model.lm_encoder.rnn)
                else:
                    apply_weight_norm(model.lm_encoder)
                model.lm_encoder.load_state_dict(sd)
                remove_weight_norm(model)

        if self.args.neurons > 0:
            print('WARNING. Setting neurons %s' % str(self.args.neurons))
            model.set_neurons(self.args.neurons)
        return model

    def __init__(self):
        self.args = Args()
        self.model = self.get_model()

    def _classify(self, text):
        # Make sure to set *both* parts of the model to .eval() mode.
        self.model.lm_encoder.eval()
        self.model.classifier.eval()
        # Initialize data, append results
        stds = np.array([])
        labels = np.array([])
        label_probs = np.array([])
        first_label = True
        heads_per_class = self.args.heads_per_class

        def get_batch(batch):
            text = batch['text'][0]
            timesteps = batch['length']
            labels = batch['label']
            text = Variable(text).long()
            timesteps = Variable(timesteps).long()
            labels = Variable(labels).long()
            if self.args.max_seq_len is not None:
                text = text[:, :self.args.max_seq_len]
                timesteps = torch.clamp(timesteps, max=self.args.max_seq_len)
            if self.args.cuda:
                text, timesteps, labels = text.cuda(), timesteps.cuda(), labels.cuda()
            return text.t(), labels, timesteps - 1

        def get_outs(text_batch, length_batch):
            if self.args.model.lower() == 'transformer':
                class_out, (lm_or_encoder_out, state) = self.model(text_batch, length_batch, self.args.get_hidden)
            else:
                self.model.lm_encoder.rnn.reset_hidden(self.args.batch_size)
                for _ in range(1 + self.args.num_hidden_warmup):
                    class_out, (lm_or_encoder_out, state) = self.model(text_batch, length_batch, self.args.get_hidden)
            if self.args.use_softmax and self.args.heads_per_class == 1:
                class_out = F.softmax(class_out, -1)
            return class_out, (lm_or_encoder_out, state)

        tstart = start = time.time()
        n = 0
        len_ds = len(text)
        with torch.no_grad():
            for i, data in enumerate(text):
                text_batch, labels_batch, length_batch = get_batch(data)
                size = text_batch.size(1)
                n += size
                # get predicted probabilities given transposed text and lengths of text
                probs, _ = get_outs(text_batch, length_batch)
                #            probs = model(text_batch, length_batch)
                if first_label:
                    first_label = False
                    labels = []
                    label_probs = []
                    if heads_per_class > 1:
                        stds = []
                # Save variances, and predictions
                # TODO: Handle multi-head [multiple classes out]
                if heads_per_class > 1:
                    _, probs, std, preds = probs
                    stds.append(std.data.cpu().numpy())
                else:
                    probs, preds = probs
                    if self.args.use_softmax:
                        probs = F.softmax(probs, -1)
                labels.append(preds.data.cpu().numpy())
                label_probs.append(probs.data.cpu().numpy())

                num_char = length_batch.sum().item()

                end = time.time()
                elapsed_time = end - start
                total_time = end - tstart
                start = end

                s_per_batch = total_time / (i + 1)

        if not first_label:
            labels = (np.concatenate(labels))  # .flatten())
            label_probs = (np.concatenate(label_probs))  # .flatten())
            if heads_per_class > 1:
                stds = (np.concatenate(stds))
            else:
                stds = np.zeros_like(labels)
        print('%0.3f seconds to transform %d examples' %
              (time.time() - tstart, n))
        return labels, label_probs, stds

    def classify(self, path='data.csv'):
        self.args.data[0] = path
        (train_data, val_data, test_data), tokenizer = self.get_data()

        ypred, yprob, ystd = self._classify(train_data)
        return ypred, yprob, ystd
