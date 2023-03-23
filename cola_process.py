from transformers import AutoTokenizer
import pandas as pd
import numpy as np

def process_data(df, model_name):
    res = {0:[], 1:[], 'type':[], 'ungram':[], 'watch_points':[]}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for i in range(df.shape[0] -1):
        t1, l1 = df.loc[i]['text'], df.loc[i]['label']
        t2, l2 = df.loc[i+1]['text'], df.loc[i+1]['label']
        if t1 == t2:
            continue
        tt1, tt2 = tokenizer(t1, add_special_tokens=False)['input_ids'], tokenizer(t2, add_special_tokens=False)['input_ids']
        set1, set2 = set(tt1), set(tt2)
        if l1 == l2 == 1:
            res['ungram'].append(2)
        elif l1 != l2:
            res['ungram'].append(1)
        else:
            res['ungram'].append(0)
        if len(set1 - set2) == len(set2 - set1) == 1 and len(tt2) == len(tt1):
            res['type'].append('swap')
            if l2 == 1:
                res[0].append(t1)
                res[1].append(t2)
                res['watch_points'].append(np.argwhere(np.array(tt1)==list(set1 - set2)[0])[0])
                
            else:
                res[0].append(t2)
                res[1].append(t1)
                res['watch_points'].append(np.argwhere(np.array(tt2)==list(set2 - set1)[0])[0])

        elif len(set1 - set2) == 0 and len(set2 - set1) == 1 and len(tt2) - len(tt1) == 1:
            res['watch_points'].append(np.argwhere(np.array(tt2)==list(set2 - set1)[0])[0])
            if l2 == 1:
                res['type'].append('deletion')
                res[0].append(t1)
                res[1].append(t2)
                
            else:
                res['type'].append('insertion')
                res[0].append(t2)
                res[1].append(t1)
        elif len(set1 - set2) == 1 and len(set2 - set1) == 0 and len(tt1) - len(tt2) == 1:
            res['watch_points'].append(np.argwhere(np.array(tt1)==list(set1 - set2)[0])[0])
            if l2 == 1:
                res['type'].append('insertion')
                res[0].append(t1)
                res[1].append(t2)
            else:
                res['type'].append('deletion')
                res[0].append(t2)
                res[1].append(t1)
        elif len(set1 - set2) ==  len(set2 - set1) == 0 and len(tt1) == len(tt2):
            res['watch_points'].append(np.argwhere(np.array(tt1)!=np.array(tt2))[0])
            res['type'].append('order')
            if l2 == 1:
                res[0].append(t1)
                res[1].append(t2)
            else:
                res[0].append(t2)
                res[1].append(t1)
        else:
            res['ungram'].pop()
    samples = pd.DataFrame(res)
    return samples

tr_df = pd.read_csv('cola_train.csv', sep='\t', names=['source', 'label', 'ast', 'text'])
samples= process_data(tr_df, 'microsoft/deberta-v3-base')
samples.to_csv('processed_data.csv', sep = '\t')