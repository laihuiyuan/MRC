import re
import sys
import json
import random
random.seed(42)
from collections import Counter


answers = []
with open('coc_outputs/' + sys.argv[1] + '-fs-coc0-math500.json','r') as f:
    f = json.load(f)
    for line in f:
        # answer = float(line['answer'].replace(',',''))
        answer = line['answer']
        answers.append(answer)

predictions = []
for order in ['0','1','2','3','4','5','6','7']:
    temp = []
    with open('coc_outputs/' + sys.argv[1] + '-fs-coc{}-math500.json'.format(order),'r') as f:
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

num = 0
for i in range(len(predic)):
    random.shuffle(predic[i])
    pred = max(predic[i], key=predic[i].count)
    if pred==answers[i]:
        num += 1   
print('[info] | Accuracy: %.4f | Correct %d | Total %d' % (
float(num / len(answers)), num, len(answers)))

#score = 0
#for line in predic:
#    counter = Counter(line)
#    max_count = max(counter.values())
#    score += max_count/len(line)
#print('[info] | ILC: %.4f' %(score/len(predic)))

