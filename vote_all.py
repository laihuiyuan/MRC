import re
import sys
import json
import random
random.seed(42)
from collections import Counter

from utils import (
    get_final_result_gsm8k, 
    get_final_result_math, 
    get_csqa_match,
    _strip_string
)

mgsm_list = []
with open('data/gsm8k.json', 'r', encoding='utf-8') as f1, open('data/mgsm/mgsm_en.json', 'r', encoding='utf-8') as f2:
    f1 = json.load(f1)
    f2 = json.load(f2)
    questions = []
    for line in f1:
        questions.append(line['question'][:20])
    for i, line in enumerate(f2):
        try:
            x = questions.index(line['question'][:20])
            mgsm_list.append(x)
        except:
            continue
#print(mgsm_list[:10])


answers = []
with open('la_outputs/' + sys.argv[3] + '-fs-mgsm-en.json','r') as f:
    f = json.load(f)
    for line in f:
        answer = float(line['answer'].replace(',',''))
        answers.append(answer)


predictions = []
#for order in ['1','2','3','4']:
for order in ['0','1','2','3','4','5','6','7']:
    temp = []
    with open('or_outputs/' + sys.argv[1] + '-fs-or{}-gsm8k.json'.format(order),'r') as f:
        f = json.load(f)
        for i in mgsm_list:
            temp.append(f[i]['prediction'])
    predictions.append(temp)


#temp = []
#with open('rw_outputs/' + sys.argv[2] + '-fs-or-gsm8k.json','r') as f:
#    f = json.load(f)
#    for i in mgsm_list:
#        temp.append(f[i]['prediction'])
#    predictions.append(temp)

temp = []
with open('rw_outputs/' + sys.argv[2] + '-fs-rw-gsm8k.json','r') as f:
    f = json.load(f)
    for i in mgsm_list:
        temp.append(f[i]['prediction'])
    predictions.append(temp)


temp = []
with open('rw_outputs/' + sys.argv[2] + '-fs-so-gsm8k.json','r') as f:
    f = json.load(f)
    for i in mgsm_list:
        temp.append(f[i]['prediction'])
    predictions.append(temp)

    
temp = []
with open('rs_outputs/' + sys.argv[2] + '-fs-so-gsm8k.json','r') as f:
    f = json.load(f)
    for i in mgsm_list:
        temp.append(f[i]['prediction'])
    predictions.append(temp)


temp = []
with open('rs_outputs/' + sys.argv[2] + '-fs-sb-gsm8k.json','r') as f:
    f = json.load(f)
    for i in mgsm_list:
        temp.append(f[i]['prediction'])
    predictions.append(temp)


#for lang in ['fr','en','ja','zh','es']:
for lang in ['bn','de','en','es','fr','ja','ru','sw','te','th','zh']:
    temp = []
    with open('la_outputs/' + sys.argv[3] + '-fs-mgsm-{}.json'.format(lang),'r') as f:
        f = json.load(f)
        for line in f:
            temp.append(line['prediction'])
    predictions.append(temp)

#for line in predictions:
#    print(len(line))

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
#print('[info] | ALL: %.4f' %(score/len(predic)))

