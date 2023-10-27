import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask

from pcgrad import PCGrad

TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        ### TODO
        #raise NotImplementedError
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dropout_sts = torch.nn.Dropout(0.5) # higher dropout for sts to combat overfitting
        # self.linear_sentiment = torch.nn.Sequential(
        #     torch.nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES))
        self.rnn = torch.nn.RNN(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE, 2)
        self.linear_sentiment = torch.nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        # yoko was told the second dimension here should be 1
        # self.linear2 = torch.nn.Linear(2*BERT_HIDDEN_SIZE, 2) # paraphrase detection
        # self.linear3 = torch.nn.Linear(2*BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES+1) # similarity
        # self.linear_para = torch.nn.Sequential(
        #     torch.nn.Linear(2*BERT_HIDDEN_SIZE, 2*BERT_HIDDEN_SIZE),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(2*BERT_HIDDEN_SIZE,1))
        self.linear_para = torch.nn.Linear(2*BERT_HIDDEN_SIZE,1)
        # self.linear_sts = torch.nn.Sequential(
        #     torch.nn.Linear(2*BERT_HIDDEN_SIZE, 2*BERT_HIDDEN_SIZE),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(2*BERT_HIDDEN_SIZE,1))
        self.linear_sts = torch.nn.Linear(2*BERT_HIDDEN_SIZE,1)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        
        pooled_rep = self.bert.forward(input_ids, attention_mask)['pooler_output']
        return pooled_rep


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        pooled_rep = self.forward(input_ids, attention_mask)
        hidden = torch.zeros(2, BERT_HIDDEN_SIZE)
        out, hidden = self.rnn(pooled_rep, hidden)
        logits = self.linear_sentiment(self.dropout(out))
        return logits
        
        
         #raise NotImplementedError


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        ### TODO
        pooled_rep_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_rep_2 = self.forward(input_ids_2, attention_mask_2)
        # combine embeddings
        final_embed = torch.cat((pooled_rep_1, pooled_rep_2), dim=1)
        logits = self.linear_para(self.dropout(final_embed))
        return logits
        
                
        #raise NotImplementedError


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        pooled_rep_1 = self.forward(input_ids_1, attention_mask_1)
        pooled_rep_2 = self.forward(input_ids_2, attention_mask_2)
        # combine embeddings
        final_embed = torch.cat((pooled_rep_1, pooled_rep_2), dim=1)
        logits = self.linear_sts(self.dropout_sts(final_embed))
        return logits
        #raise NotImplementedError




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        #'optim': optimizer._optim.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


## Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_dataloader = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_dataloader = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_dataloader, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_dataloader.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_dataloader, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_dataloader.collate_fn)
    
    # load quora data
    quora_train_dataloader = SentencePairDataset(para_train_data, args)
    quora_dev_dataloader = SentencePairDataset(para_dev_data, args)
    
    quora_train_dataloader = DataLoader(quora_train_dataloader, shuffle=True, batch_size=args.batch_size,collate_fn=quora_train_dataloader.collate_fn)
    quora_dev_dataloader = DataLoader(quora_dev_dataloader, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=quora_dev_dataloader.collate_fn)
                                    
    # load STS data
    sts_train_dataloader = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_dataloader = SentencePairDataset(sts_dev_data, args, isRegression=True)
    
    sts_train_dataloader = DataLoader(sts_train_dataloader, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_dataloader.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_dataloader, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_dataloader.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    # wrap your favorite optimizer
    #optimizer = PCGrad( AdamW(model.parameters(), lr=lr) )
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_sst, batch_quora, batch_sts in tqdm( zip(sst_train_dataloader, quora_train_dataloader, sts_train_dataloader),
        desc=f'train-{epoch}', disable=TQDM_DISABLE):
        
            # quora dataset
            bq_ids_1, bq_mask_1, bq_ids_2, bq_mask_2, bq_labels = (batch_quora['token_ids_1'], batch_quora['attention_mask_1'],
            batch_quora['token_ids_2'], batch_quora['attention_mask_2'], batch_quora['labels'])
            bq_ids_1 = bq_ids_1.to(device)
            bq_ids_2 = bq_ids_2.to(device)
            bq_mask_1 = bq_mask_1.to(device)
            bq_mask_2 = bq_mask_2.to(device)
            bq_labels = bq_labels.to(device)
            
            # sst dataset
            bs_ids, bs_mask, bs_labels = (batch_sst['token_ids'], batch_sst['attention_mask'], batch_sst['labels'])
            bs_ids = bs_ids.to(device)
            bs_mask = bs_mask.to(device)
            bs_labels = bs_labels.to(device)
            
            # sts dataset
            bt_ids_1, bt_mask_1, bt_ids_2, bt_mask_2, bt_labels = (batch_sts['token_ids_1'], batch_sts['attention_mask_1'],
            batch_sts['token_ids_2'], batch_sts['attention_mask_2'], batch_sts['labels'])
            bt_ids_1 = bt_ids_1.to(device)
            bt_ids_2 = bt_ids_2.to(device)
            bt_mask_1 = bt_mask_1.to(device)
            bt_mask_2 = bt_mask_2.to(device)
            bt_labels = bt_labels.to(device)

            optimizer.zero_grad()
            
            logits_quora = model.predict_paraphrase(bq_ids_1, bq_mask_1, bq_ids_2, bq_mask_2)
            logits_sst = model.predict_sentiment(bs_ids, bs_mask)
            logits_sts = model.predict_similarity(bt_ids_1, bt_mask_1, bt_ids_2, bt_mask_2)
            
            loss_sentiment = F.cross_entropy(logits_sst, bs_labels.view(-1), reduction='sum') / args.batch_size
            
            #binary cross entropy used for now
            loss_paraphrase = F.binary_cross_entropy(logits_quora.sigmoid().view(-1), bq_labels.float(), reduction='sum') / args.batch_size
            
            loss_similarity = F.mse_loss(logits_sts.view(-1), bt_labels.float(), reduction='sum') / args.batch_size
            
            loss = loss_paraphrase + loss_sentiment + loss_similarity
            
            #losses = [loss_sentiment, loss_paraphrase, loss_similarity] # a list of per-task losses
            loss.backward()
            #optimizer.pc_backward(losses)
            optimizer.step()

            train_loss += loss
            num_batches += 1


        train_loss = train_loss / (num_batches)
        
        train_para_acc, _, _, train_sentiment_acc, _, _, train_sts_corr, _, _ = model_eval_multitask(sst_train_dataloader, quora_train_dataloader, sts_train_dataloader, model, device)
        
        dev_para_acc, _, _, dev_sentiment_acc, _, _, dev_sts_corr, _, _ = model_eval_multitask(sst_dev_dataloader, quora_dev_dataloader, sts_dev_dataloader, model, device)
                         
                         
        # train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
        # dev_acc, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        
        # change to comibned dev accuracies?
        if (dev_sentiment_acc + dev_para_acc + dev_sts_corr)/3 > best_dev_acc:
            best_dev_acc = (dev_sentiment_acc+dev_para_acc + dev_sts_corr)/3
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}, train para acc :: {train_para_acc:.3f}, dev para acc :: {dev_para_acc:.3f}")
        print(f"train_sentiment_acc :: {train_sentiment_acc:.3f}, dev_sentiment_acc :: {dev_sentiment_acc:.3f}")
        print(f"train_sts_corr :: {train_sts_corr:.3f}, dev_sts_corr :: {dev_sts_corr:.3f}")



def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
