import os
#os.environ['CUDA_VISIBLE_DEVICES']=str(0)
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd


#Answer.question
def get_data(path):
    tab = pd.read_csv(path, sep=',') 
    rel = tab[['Answer.question', 'Answer.ref', 'Answer.t1', 'Answer.t2', 'Answer.tooFast']]
    filtered = rel[rel['Answer.tooFast'] < 10] 
    return filtered

parser = argparse.ArgumentParser(description='Train network.')
parser.add_argument('--gender', type=str, help='gender of model')

args = parser.parse_args()
gender = args.gender

directory = './data'
data_1 = get_data(os.path.join(directory,'batchR1_' + gender + '.csv'))
data_2 = get_data(os.path.join(directory,'batchR2_' + gender + '.csv'))
data_3 = get_data(os.path.join(directory,'batchR3_' + gender + '.csv'))
data_4 = get_data(os.path.join(directory,'batchR4_' + gender + '.csv'))
data_5 = get_data(os.path.join(directory,'batchR5_' + gender + '.csv'))
data_6 = get_data(os.path.join(directory,'batchR6_' + gender + '.csv'))
data_7 = get_data(os.path.join(directory,'batchR7_' + gender + '.csv'))
data_8 = get_data(os.path.join(directory,'batchR8_' + gender + '.csv'))
data_9 = get_data(os.path.join(directory,'batchR9_' + gender + '.csv'))

filtered = pd.concat([data_1], axis=0, ignore_index=True)#, data_2, data_3, data_4, data_5, data_6, data_7, data_8, data_9], axis=0, ignore_index=True)
nitems = len(filtered.index)
#print("nitems", nitems)


apns = [] #anchor-positive-negative #TODO: check processing of images
nitems = -1
for i in range(len(filtered.index)):
    choices = filtered.iloc[i]['Answer.question'].split()
    ref = filtered.iloc[i]['Answer.ref'].split()
    t1 = filtered.iloc[i]['Answer.t1'].split()
    t2 = filtered.iloc[i]['Answer.t2'].split()
    for c in range(len(choices)):
        ref1 = int(ref[c]) - 1
        c1 = int(t1[c]) - 1
        c2 = int(t2[c]) - 1
        nitems = max(nitems, ref1 + 1, c1 + 1, c2 + 1)
        if ref[c] == 't1':
            apns.append((ref1, c1, c2))
        else:
            apns.append((ref1, c2, c1)) #add tuples in the format (ref, chosen, not chosen)
print(len(apns))
np.random.shuffle(apns)
hn = int(len(apns)*.9)

train_apns = apns[:hn]
test_apns = apns[hn:]

np.savetxt('train90.txt', np.array(train_apns).astype(int), fmt='%i')
np.savetxt('test10.txt', np.array(test_apns).astype(int), fmt='%i')
