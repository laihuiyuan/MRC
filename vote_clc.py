import re
import sys
import json
import random
random.seed(42)
from collections import Counter

from utils import (
    get_final_result_gsm8k, 
    get_final_result_math, 
    _strip_string
)

answers = []
with open('clc_outputs/' + sys.argv[1] + '-fs-math500-en.json','r') as f:
    f = json.load(f)
    for line in f:
        # answer = float(line['answer'].replace(',',''))
        answer = line['answer']
        answers.append(answer)

predictions = []
#for lang in ['de','en','es','fr','ja','ru','th','zh']:
for lang in ['bn','de','en','es','fr','ja','ru','sw','te','th','zh']:
    temp = []
    with open('clc_outputs/' + sys.argv[1] + '-fs-math500-{}.json'.format(lang),'r') as f:
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
#print('[info] | CLC: %.4f' %(score/len(predic)))

