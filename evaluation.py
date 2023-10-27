#!/usr/bin/env python3

'''
Model evaluation functions.

When training your multitask model, you will find it useful to run
model_eval_multitask to be able to evaluate your model on the 3 tasks in the
development set.

Before submission, your code needs to call test_model_multitask(args, model, device) to generate
your predictions. We'll evaluate these predictions against our labels on our end,
which is how the leaderboard will be updated.
The provided test_model() function in multitask_classifier.py **already does this for you**,
so unless you change it you shouldn't need to call anything from here
explicitly aside from model_eval_multitask.
'''

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, precision_score, balanced_accuracy_score
from tqdm import tqdm
import numpy as np

from datasets import load_multitask_data, load_multitask_test_data, \
    SentenceClassificationDataset, SentenceClassificationTestDataset, \
    SentencePairDataset, SentencePairTestDataset


TQDM_DISABLE = True

# Evaluate a multitask model for accuracy.on SST only.
def model_eval_sst(dataloader, model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'],batch['attention_mask'],  \
                                                        batch['labels'], batch['sents'], batch['sent_ids']

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, y_pred, y_true, sents, sent_ids

# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():
        para_y_true, para_y_pred, para_sent_ids = [], [], []
        sts_y_true, sts_y_pred, sts_sent_ids = [], [], []
        sst_y_true, sst_y_pred, sst_sent_ids = [], [], []
        '''
        for batch_sst, batch_quora, batch_sts in tqdm( zip(sentiment_dataloader, paraphrase_dataloader, sts_dataloader),
        desc=f'eval', disable=TQDM_DISABLE):
        
            # quora dataset
            bq_ids_1, bq_mask_1, bq_ids_2, bq_mask_2, bq_labels, bq_sent_ids = (batch_quora['token_ids_1'], batch_quora['attention_mask_1'],
            batch_quora['token_ids_2'], batch_quora['attention_mask_2'], batch_quora['labels'], batch_quora['sent_ids'])
            bq_ids_1 = bq_ids_1.to(device)
            bq_ids_2 = bq_ids_2.to(device)
            bq_mask_1 = bq_mask_1.to(device)
            bq_mask_2 = bq_mask_2.to(device)
            
            para_logits = model.predict_paraphrase(bq_ids_1, bq_mask_1, bq_ids_2, bq_mask_2)
            para_y_hat = para_logits.sigmoid().round().flatten().cpu().numpy()
            bq_labels = bq_labels.flatten().cpu().numpy()

            para_y_pred.extend(para_y_hat)
            para_y_true.extend(bq_labels)
            para_sent_ids.extend(bq_sent_ids)
            
            # sst dataset
            bs_ids, bs_mask, bs_labels, bs_sent_ids = (batch_sst['token_ids'], batch_sst['attention_mask'], batch_sst['labels'], batch_sst['sent_ids'])

            bs_ids = bs_ids.to(device)
            bs_mask = bs_mask.to(device)

            sst_logits = model.predict_sentiment(bs_ids, bs_mask)
            sst_y_hat = sst_logits.argmax(dim=-1).flatten().cpu().numpy()
            bs_labels = bs_labels.flatten().cpu().numpy()

            sst_y_pred.extend(sst_y_hat)
            sst_y_true.extend(bs_labels)
            sst_sent_ids.extend(bs_sent_ids)
            
            # sts dataset
            bt_ids_1, bt_mask_1, bt_ids_2, bt_mask_2, bt_labels, bt_sent_ids = (batch_sts['token_ids_1'], batch_sts['attention_mask_1'],
            batch_sts['token_ids_2'], batch_sts['attention_mask_2'], batch_sts['labels'], batch_sts['sent_ids'])
            bt_ids_1 = bt_ids_1.to(device)
            bt_ids_2 = bt_ids_2.to(device)
            bt_mask_1 = bt_mask_1.to(device)
            bt_mask_2 = bt_mask_2.to(device)
            
            sts_logits = model.predict_similarity(bt_ids_1, bt_mask_1, bt_ids_2, bt_mask_2)
            sts_y_hat = sts_logits.flatten().cpu().numpy()
            bt_labels = bt_labels.flatten().cpu().numpy()

            sts_y_pred.extend(sts_y_hat)
            sts_y_true.extend(bt_labels)
            sts_sent_ids.extend(bt_sent_ids)
            
        '''
        # Evaluate paraphrase detection.
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_y_true.extend(b_labels)
            para_sent_ids.extend(b_sent_ids)

        paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))
        paraphrase_balacc = balanced_accuracy_score(para_y_true, para_y_pred)
        #paraphrase_precision = precision_score(para_y_true, para_y_pred, average='macro')
        #paraphrase_recall = recall_score(para_y_true, para_y_pred, average='macro')
        paraphrase_F1 = f1_score(para_y_true, para_y_pred, average='macro')
        #precision, recall and F1 score
        

        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []
        
        # Evaluate semantic textual similarity.
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['labels'], batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)
        pearson_mat = np.corrcoef(sts_y_pred, sts_y_true)
        sts_corr = pearson_mat[1][0]


        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_y_true.extend(b_labels)
            sst_sent_ids.extend(b_sent_ids)
        
        
        paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))
        paraphrase_balacc = balanced_accuracy_score(para_y_true, para_y_pred)
        #paraphrase_precision = precision_score(para_y_true, para_y_pred, average='macro')
        #paraphrase_recall = recall_score(para_y_true, para_y_pred, average='macro')
        paraphrase_F1 = f1_score(para_y_true, para_y_pred, average='macro')
        #precision, recall and F1 score
        
        pearson_mat = np.corrcoef(sts_y_pred, sts_y_true)
        sts_corr = pearson_mat[1][0]

        sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))
        sentiment_balacc = balanced_accuracy_score(sst_y_true, sst_y_pred)
        #sentiment_precision = precision_score(sst_y_true, sst_y_pred, average='macro')
        #sentiment_recall = recall_score(sst_y_true, sst_y_pred, average='macro')
        sentiment_F1 = f1_score(sst_y_true, sst_y_pred, average='macro')

        print(f'Paraphrase detection accuracy: {paraphrase_accuracy:.3f}')
        print(f'Paraphrase detection F1 score: {paraphrase_F1:.3f}')
        print(f'Paraphrase detection balanced accuracy: {paraphrase_balacc:.3f}')
        #{paraphrase_recall:.3f}, {paraphrase_precision:.3f}')
        print(f'Sentiment classification accuracy: {sentiment_accuracy:.3f}')
        print(f'Sentiment classification F1 score: {sentiment_F1:.3f}')
        print(f'Sentiment classification balanced accuracy: {sentiment_balacc:.3f}')
        #{sentiment_recall:.3f}, {sentiment_precision:.3f}')
        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return (paraphrase_accuracy, para_y_pred, para_sent_ids,
                sentiment_accuracy,sst_y_pred, sst_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids)

# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_test_multitask(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():

        para_y_pred = []
        para_sent_ids = []
        # Evaluate paraphrase detection.
        for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.sigmoid().round().flatten().cpu().numpy()

            para_y_pred.extend(y_hat)
            para_sent_ids.extend(b_sent_ids)


        sts_y_pred = []
        sts_sent_ids = []


        # Evaluate semantic textual similarity.
        for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
             b_ids2, b_mask2,
             b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                          batch['token_ids_2'], batch['attention_mask_2'],
                          batch['sent_ids'])

            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
            y_hat = logits.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_sent_ids.extend(b_sent_ids)


        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids, b_mask, b_sent_ids = batch['token_ids'], batch['attention_mask'],  batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model.predict_sentiment(b_ids, b_mask)
            y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()

            sst_y_pred.extend(y_hat)
            sst_sent_ids.extend(b_sent_ids)

        return (para_y_pred, para_sent_ids,
                sst_y_pred, sst_sent_ids,
                sts_y_pred, sts_sent_ids)


def test_model_multitask(args, model, device):
        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args) 

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, dev_sts_corr, \
            dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_para_y_pred, test_para_sent_ids, test_sst_y_pred, \
            test_sst_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")
