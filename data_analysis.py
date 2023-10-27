'''
Analyze data by label for each dataset.
'''
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch

# Similarity (STS)
sts_preds = np.zeros(864)
sts_true = np.zeros(864)
with open("predictions/sts-dev-output.csv") as sts_preds_csv:
    preds_reader = csv.reader(sts_preds_csv, delimiter=',')
    line_count = -1
    for row in preds_reader:
        if line_count == -1:
            line_count += 1
            continue
        else:
            sts_preds[line_count] = round(float(row[1]))
            line_count += 1
    print(f"Processed {line_count + 1} rows of STS Predictions.")

with open("data/sts-dev.csv") as sts_true_csv:
    true_reader = csv.reader(sts_true_csv, delimiter='\t')
    line_count = -1
    for row in true_reader:
        if line_count == -1:
            line_count += 1
            continue
        else:
            sts_true[line_count] = round(float(row[4]))
            line_count += 1
    print(f"Processed {line_count + 1} rows of STS True Labels.")

sts0_preds = [] # 0 - 0.5 true
sts0_acc = 0
sts1_preds = [] # 0.5 - 1.5 true
sts1_acc = 0
sts2_preds = [] # 1.5 - 2.5 true
sts2_acc = 0
sts3_preds = [] # 2.5 - 3.5 true
sts3_acc = 0
sts4_preds = [] # 3.5 - 4.5 true
sts4_acc = 0
sts5_preds = [] # 4.5 - 5.0 true
sts5_acc = 0

for pred, true in zip(sts_preds, sts_true):
    # how is accuracy measured for similarity? what tolerance?
    if true >= 0 and true < 0.5:
        sts0_preds.append(pred)
        if pred == true:
            sts0_acc += 1
    elif true >= 0.5 and true < 1.5:
        sts1_preds.append(pred)
        if pred == true:
            sts1_acc += 1
    elif true >= 1.5 and true < 2.5:
        sts2_preds.append(pred)
        if pred == true:
            sts2_acc += 1
    elif true >= 2.5 and true < 3.5:
        sts3_preds.append(pred)
        if pred == true:
            sts3_acc += 1
    elif true >= 3.5 and true < 4.5:
        sts4_preds.append(pred)
        if pred == true:
            sts4_acc += 1
    else: # true >= 4.5
        sts5_preds.append(pred)
        if pred == true:
            sts5_acc += 1

print("STS total accuracy: ", np.mean(sts_preds == sts_true))
print("STS accuracies by label:")
print(f"0.0 to 0.5 true label: {sts0_acc / len(sts0_preds)} (average over {len(sts0_preds)} values)")
print(f"0.5 to 1.5 true label: {sts1_acc / len(sts1_preds)} (average over {len(sts1_preds)} values)")
print(f"1.5 to 2.5 true label: {sts2_acc / len(sts2_preds)} (average over {len(sts2_preds)} values)")
print(f"2.5 to 3.5 true label: {sts3_acc / len(sts3_preds)} (average over {len(sts3_preds)} values)")
print(f"3.5 to 4.5 true label: {sts4_acc / len(sts4_preds)} (average over {len(sts4_preds)} values)")
print(f"4.5 to 5.0 true label: {sts5_acc / len(sts5_preds)} (average over {len(sts5_preds)} values)")

sts_class_accs = [sts0_acc / len(sts0_preds), sts1_acc / len(sts1_preds), sts2_acc / len(sts2_preds), \
    sts3_acc / len(sts3_preds), sts4_acc / len(sts4_preds), sts5_acc / len(sts5_preds)]
plt.figure()
plt.bar(['0','1','2','3','4','5'], sts_class_accs, color='#8C1515')
plt.xlabel("Similarity (STS) True Label")
plt.ylabel("Class Accuracy")
plt.savefig("figures/sts_class_acc.jpg")


# Sentiment (SST)
print("-------------------")
sst_preds = np.zeros(1102)
sst_true = np.zeros(1102)
with open("predictions/sst-dev-output.csv") as sst_preds_csv:
    preds_reader = csv.reader(sst_preds_csv, delimiter=',')
    line_count = -1
    for row in preds_reader:
        if line_count == -1:
            line_count += 1
            continue
        else:
            sst_preds[line_count] = int(row[1])
            line_count += 1
    print(f"Processed {line_count + 1} rows of SST Predictions.")

with open("data/ids-sst-dev.csv") as sst_true_csv:
    true_reader = csv.reader(sst_true_csv, delimiter='\t')
    line_count = -1
    for row in true_reader:
        if line_count == -1:
            line_count += 1
            continue
        else:
            sst_true[line_count] = int(row[3])
            line_count += 1
    print(f"Processed {line_count + 1} rows of SST True Labels.")

sst0_preds = []
sst0_acc = 0
sst1_preds = []
sst1_acc = 0
sst2_preds = []
sst2_acc = 0
sst3_preds = []
sst3_acc = 0
sst4_preds = []
sst4_acc = 0

for pred, true in zip(sst_preds, sst_true):
    # how is accuracy measured for similarity? what tolerance?
    if true == 0:
        sst0_preds.append(pred)
        if true == pred:
            sst0_acc += 1
    elif true == 1:
        sst1_preds.append(pred)
        if true == pred:
            sst1_acc += 1
    elif true == 2:
        sst2_preds.append(pred)
        if true == pred:
            sst2_acc += 1
    elif true == 3:
        sst3_preds.append(pred)
        if true == pred:
            sst3_acc += 1
    elif true == 4:
        sst4_preds.append(pred)
        if true == pred:
            sst4_acc += 1

print("SST total accuracy: ", np.mean(sst_preds == sst_true))
print("SST accuracies by label:")
print(f"0 true label: {sst0_acc / len(sst0_preds)} (average over {len(sst0_preds)} values)")
print(f"1 true label: {sst1_acc / len(sst1_preds)} (average over {len(sst1_preds)} values)")
print(f"2 true label: {sst2_acc / len(sst2_preds)} (average over {len(sst2_preds)} values)")
print(f"3 true label: {sst3_acc / len(sst3_preds)} (average over {len(sst3_preds)} values)")
print(f"4 true label: {sst4_acc / len(sst4_preds)} (average over {len(sst4_preds)} values)")

sst_class_accs = [sst0_acc / len(sst0_preds), sst1_acc / len(sst1_preds), sst2_acc / len(sst2_preds),\
    sst3_acc / len(sst3_preds), sst4_acc / len(sst4_preds)]
plt.figure()
plt.bar(['0','1','2','3','4'], sst_class_accs, color='#8C1515')
plt.xlabel("Sentiment (SST) True Label")
plt.ylabel("Class Accuracy")
plt.savefig("figures/sst_class_acc.jpg")


# Paraphrase
print("--------------")
para_preds = np.zeros(20215)
para_true = np.zeros(20215)
with open("predictions/para-dev-output.csv") as para_preds_csv:
    preds_reader = csv.reader(para_preds_csv, delimiter=',')
    line_count = -1
    for row in preds_reader:
        if line_count == -1:
            line_count += 1
            continue
        else:
            para_preds[line_count] = round(float(row[1]))
            line_count += 1
    print(f"Processed {line_count + 1} rows of Paraphrase Predictions.")

with open("data/quora-dev.csv") as para_true_csv:
    true_reader = csv.reader(para_true_csv, delimiter='\t')
    line_count = -1
    for row in true_reader:
        if line_count == -1:
            line_count += 1
            continue
        else:
            if row[4] != '':
                para_true[line_count] = float(row[4])
            else:
                para_true[line_count] = 0.5
            line_count += 1
    print(f"Processed {line_count + 1} rows of Paraphrase True Labels.")

para_preds = para_preds

para0_preds = []
para0_acc = 0
para1_preds = []
para1_acc = 0

for pred, true in zip(para_preds, para_true):
    # how is accuracy measured for similarity? what tolerance?
    if true == 0:
        para0_preds.append(pred)
        if true == pred:
            para0_acc += 1
    elif true == 1:
        para1_preds.append(pred)
        if true == pred:
            para1_acc += 1

print("Para total accuracy: ", np.mean(para_preds == para_true))
print("Para accuracies by label:")
print(f"0 true label: {para0_acc / len(para0_preds)} (average over {len(para0_preds)} values)")
print(f"1 true label: {para1_acc / len(para1_preds)} (average over {len(para1_preds)} values)")

plt.figure()
plt.bar(['0','1'],[para0_acc / len(para0_preds), para1_acc / len(para1_preds)], color='#8C1515')
plt.xlabel("Paraphrase True Label")
plt.ylabel("Class Accuracy")
plt.savefig("figures/para_class_acc.jpg")


# baseline to Final Architecture scores
x = ["Paraphrase", "SST", "STS"]
y_base = [0.667, 0.347, 0.254]
y_final = [0.728, 0.516, 0.438]
plt.figure()
plt.plot(x, y_base, color = '#8C1515',label="Baseline")
plt.plot(x, y_final, color = '#175E53',label="Final Model")
plt.xlabel("Task")
plt.ylabel("Score")
plt.legend()
plt.savefig("figures/improvement.jpg")
