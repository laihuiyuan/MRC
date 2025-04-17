import re
import sys
import json
import random
random.seed(42)
from collections import Counter


answers = []
predictions = []

temp = []
with open('rw_outputs/' + sys.argv[1] + '-fs-or-gsm8k.json','r') as f:
    f = json.load(f)
    for line in f:
        answer = float(line['answer'].replace(',',''))
        answers.append(answer)
        temp.append(line['prediction'])
    predictions.append(temp)


temp = []
with open('rw_outputs/' + sys.argv[1] + '-fs-rw-gsm8k.json','r') as f:
    f = json.load(f)
    for line in f:
        temp.append(line['prediction'])
    predictions.append(temp)


temp = []
with open('rw_outputs/' + sys.argv[1] + '-fs-so-gsm8k.json','r') as f:
    f = json.load(f)
    for line in f:
        temp.append(line['prediction'])
    predictions.append(temp)

    
temp = []
with open('rs_outputs/' + sys.argv[1] + '-fs-so-gsm8k.json','r') as f:
    f = json.load(f)
    for line in f:
        temp.append(line['prediction'])
    predictions.append(temp)


temp = []
with open('rs_outputs/' + sys.argv[1] + '-fs-sb-gsm8k.json','r') as f:
    f = json.load(f)
    for line in f:
        temp.append(line['prediction'])
    predictions.append(temp)


predic = []
for i in range(len(predictions[0])):
    temp = []
    for line in predictions:
        temp.append(line[i])
    predic.append(temp)

#num = 0
#for i in range(len(predic)):
#    random.shuffle(predic[i])
#    pred = max(predic[i], key=predic[i].count)
#    if pred==answers[i]:
#        num += 1
#print('[info] | Accuracy: %.4f | Correct %d | Total %d' % (
#float(num / len(answers)), num, len(answers)))


score = 0
for line in predic:
    counter = Counter(line)
    max_count = max(counter.values())
    score += max_count/len(line)
print('[info] | PHC: %.4f' %(score/len(predic)))

