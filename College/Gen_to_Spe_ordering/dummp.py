from math import exp
from random import random, seed
def init_ntw(n_ip,n_hid,n_op):
  ntw = list()
  hidlyr = [{'weights': random() for i in range(n_ip-1)} for i in range(n_hid)]
  ntw.append(hidlyr)
  outlyr = [{'weights': random() for i in range(n_hid-1)} for i in range(n_op)]
  ntw.append(outlyr)
  return ntw
def activate(wt,x):
  act = wt[-1]
  for i in range(len(w)-1):
    act+= w[i]* x[i]
  return act
def transfer(act):
  val = 1.0 /(1.0 +exp(-act))
  return val
def transfer_deri(op):
  val = op * (1.0-op)
  return val
def fwdProp(ntw,row):
  ips = row
  for layer in ntw:
    new_ips = []
    for nrn in layer:
      activation = activate(nrn['weights'],ips)
      nrn['output'] = transfer(activation)
      new_ips.append(nrn['output'])
    ips = new_ips
  return ips
def bwdPropErr(ntw,exp):
  for i in reversed(range(len(ntw))):
    layer = ntw[i]
    errors=list()
    if i!= len(ntw)-1:
      for j in range(len(layer)):
        err = 0.0
        for nrn in ntw[i+1]:
          err+= nrn['weights'][j] * nrn['delta']
        errors.append(err)
    else:
      for j in range(len(layer)):
        nrn = layer[j]
        errors.append(exp[j]-nrn['output'])
    for j in range(len(layer)):
        nrn = layer[j]
        nrn['delta'] = errors[j]* transfer_deri(nrn['output'])

def update(ntw,row,lrate):
  for i in range(len(ntw)):
    ips = row[:-1]
    if i!=0:
      ips = [nrn['ouput'] for nrn in ntw[i-1]]
    for nrn in ntw[i]:
      for j in range(len(inputs)):
        neuron['weights'][j] += lrate * neuron['delta'] * ips[j]
      neuron['weights'][-1] += lrate * neuron['delta']

def train_ntw(ntw,train,lrate,epoch,n_op):
  for epoch in range(epoch):
    sumerr = 0.0
    for row in train:
      ops = fwdProp(ntw,row)
      exp = [0 for i in range(n_op)]
      exp[row[-1]] = 1
      sumerr+= sum([(exp[i]-ops[i])**2 for i in range(len(exp))])
      bwdPropErr(ntw,exp)
      update(ntw,row,lrate)
    print('>epoch = %d lrate = %.3f error = %.3f' % (epoch,lrate,sumerr))

seed(1)
dataset =[[2.7810836,2.550537003,0],
	       [1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n = len(dataset)
na = len(dataset[0])
n_ip = na + 1
n_op = len(set([row[-1] for row in dataset]))