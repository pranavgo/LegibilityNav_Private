import json
import numpy as np
import matplotlib.pyplot as plt
import numpy 
data = {} 
exp = 'corr1_results_'
# Read JSON from file
with open( str(exp + 'mpc.json'), 'r') as f:
    data[0] = json.load(f)
with open(str(exp + 'legible.json'), 'r') as f:
    data[1] = json.load(f)
with open(str(exp + 'sm_legible.json'), 'r') as f:
    data[2] = json.load(f)
# with open('results_lpsnav.json', 'r') as f:
#     data[3] = json.load(f)


pi = {key: [] for key in range(len(data))}
acc = {key: [] for key in range(len(data))}
success = {key: [] for key in range(len(data))}
pi_hist = []
acc_hist = []
success_hist = []


for key in data:
    k = 2
    for k in range(0,9,1):
        ele = data[key][k]
        if ele[0] != 0:
            pi[key].append(ele[-2])
            acc[key].append(ele[-1])
        success[key].append(ele[0])

for i in range(len(data)):
    pi_hist.append(sum(pi[i])/len(pi[i]))
    acc_hist.append(sum(acc[i])/len(acc[i]))
    success_hist.append(sum(success[i])/len(success[i]))
x = ["MPC", "ANCA","SM","LPSNAV"]
max_pi = max(pi_hist)
max_acc = max(acc_hist)
# pi_hist = [x/max_pi for x in pi_hist]
# acc_hist = [x/max_pi for x in acc_hist]

X_axis = np.arange(len(pi_hist)) 
plt.bar(X_axis - 0.2,pi_hist, color = '#FF8552', label='Path Irregularity', width=0.2)
# plt.bar(X_axis, acc_hist, color='#A4C2A5',label='Acceleration',width=0.2)
# plt.bar(X_axis + 0.2, success_hist, color='#39393A',label='Sucess rate', width=0.2)

plt.xticks(X_axis, x[:len(pi_hist)]) 
plt.legend(title='Intersection')
plt.show()