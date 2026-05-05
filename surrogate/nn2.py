#!/usr/bin/python
import numpy
numpy.bool = numpy.bool_
numpy.complex = numpy.complex_
import random
import math
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read inputs
if len(sys.argv)<2:
	quit()
pc_learn_dataset=int(sys.argv[1])
print ("percent training:"+str(pc_learn_dataset))

fname="res_pr.txt"
if len(sys.argv)>=3:
	fname=sys.argv[2]
	print("dataset filename: "+fname)

direct=1
if len(sys.argv)>=4:
	direct=int(sys.argv[3])
	print("direct (1) or inverse (0): "+str(direct))

noise=0 # absolute 
if len(sys.argv)>=5:
	noise=float(sys.argv[4])
	print("noise: "+str(noise))

output_idx=0 # one output
if len(sys.argv)>=6:
	output_idx=int(sys.argv[5])
	print("index for the function: "+str(output_idx))

nnsize=8
if len(sys.argv)>=7:
	nnsize=int(sys.argv[6])
	print("NN neurons in internal layers: "+str(nnsize))

mlp_file=""
if (len(sys.argv)>=8):
	mlp_file=sys.argv[7]
	print("mlp file: "+mlp_file)

# parameters
sets_list=[]
out_names=["u_inp","c_dur","c_inp","D","kmu","a","b","ae","tau_r"]
# read data
file=open(fname)
ll=file.read().splitlines()
# calculate a number of measurements
l=ll[0].split(",")
n_measurements=int(((len(l)-12)/3)+1)
if (len(sys.argv)>=9):
	n_measurements=int(sys.argv[8])
print("measurements:"+str(n_measurements))

inv_inp_deps=0 # inputs are concidered possibly dependent for inverse problem
if (len(sys.argv)>=10):
	inv_inp_deps=int(sys.argv[9])
print("inv_inp_deps:"+str(inv_inp_deps))

n_ep=1000
if (len(sys.argv)>=11):
	n_ep=int(sys.argv[10])
print("n iter:"+str(n_ep))

#inverse
if direct==0:
	out_size=9
	in_size=n_measurements
	if inv_inp_deps==1:
		in_size=in_size+8
	niters=n_ep
else:
# direct
	out_size=n_measurements
	in_size=9
	niters=n_ep
dataset={}
for i in range(len(ll)):
	if i==0:
		continue
	l=ll[i].split(",")
# inverse
	if direct==0:
		s=str(l[0])+" "+str(l[1])+" "+str(l[2])+" "+str(l[3])+" "+str(l[4])+" "+str(l[5])+" "+str(l[6])+" "+str(l[7])+" "+str(l[8]) # record outputs
		v=[]
		try:
			for ii in range(n_measurements):
				v.append(l[11+ii*3]) # record inputs
		except:
			for ii in range(n_measurements):
				v.append(i)
		if inv_inp_deps==1: # add outputs too
			for ii in range(9):
				if ii!=output_idx:
					v.append(l[ii])
	else:
# direct
		s=""
		try:
			for ii in range(n_measurements): # record outputs
				s=s+str(l[11+ii*3])
				if ii!=n_measurements-1:
					s=s+" "
		except:
			for ii in range(n_measurements): # record outputs
				s=s+str(i)
				if ii!=n_measurements-1:
					s=s+" "
		v=[l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8]] # record inputs
	if (s not in dataset):
		dataset[s]=[]
	dataset[s].append(v)
print(str(len(dataset))+" rows read out of "+str(len(ll)))
n=0
print("learning percent "+str(pc_learn_dataset))
for i in dataset.keys():
	if (n<=len(dataset)*pc_learn_dataset/100.0):
		sets_list.append(n)
	n=n+1
YY=[]
x_mean1=[]
x_std1=[]
y_mean1=[]
y_std1=[]
class DS:
	def __init__(self):
		global noise,YY,x_mean1,x_std1,y_mean1,y_std1
		self.X = []
		self.y = []
		YY = []
		n=0
		for i in dataset.keys():
			if n in sets_list:
				try:
					xx=[]
					# add noise
					for j in range(len(dataset[i][0])):
						xx.append(float(dataset[i][0][j])+(2.0*(random.randint(0,99)/99.0)-1.0)*noise)
					ii=i.split(" ")
					yy=[]
					for k in range(out_size):
						if k==output_idx:
							yy.append(float(ii[k]))
					self.X.append(xx)
					self.y.append(yy)
					YY.append(yy)
				except:
					pass
			n=n+1
		self.X = numpy.array(self.X)
		self.y = numpy.array(self.y)
		# Input normalization
		self.x_mean = self.X.mean(axis=0)
		self.x_std  = self.X.std(axis=0) + 1e-12
		self.y_mean = self.y.mean(axis=0)
		self.y_std  = self.y.std(axis=0) + 1e-12
		if mlp_file!="": # read normalization from files
			xfile=mlp_file.replace("mlp_torch","out_nn_norms_x_").replace(".pt",".txt")
			xn=numpy.loadtxt(xfile,delimiter=" ")
			self.x_mean=xn[0]
			self.x_std=xn[1]
			if type(self.x_mean).__name__!="ndarray":
				self.x_mean=[self.x_mean]
			if type(self.x_std).__name__!="ndarray":
				self.x_std=[self.x_std]
			yfile=mlp_file.replace("mlp_torch","out_nn_norms_y_").replace(".pt",".txt")
			yn=numpy.loadtxt(yfile,delimiter=" ")
			self.y_mean=yn[0]
			self.y_std=yn[1]
			if type(self.y_mean).__name__!="ndarray":
				self.y_mean=[self.y_mean]
			if type(self.y_std).__name__!="ndarray":
				self.y_std=[self.y_std]
		self.X = (self.X - self.x_mean) / self.x_std
		# Output normalization
		self.y = (self.y - self.y_mean) / self.y_std
		x_mean1=self.x_mean
		x_std1=self.x_std
		y_mean1=self.y_mean
		y_std1=self.y_std

class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        ls=nnsize
        self.net = nn.Sequential(
            nn.Linear(in_size, ls),
            nn.ReLU(),
            nn.Linear(ls, ls),
            nn.ReLU(),
            nn.Linear(ls, ls),
            nn.ReLU(),
            nn.Linear(ls, out_size)
        )
    def forward(self, x):
        return self.net(x)

def test(v,ds,i,out,mine):
	e=0.0
	maxe=0.0
	maxae=0.0
	avge=0.0
	ne=0
	avgv=[]
	rss=[]
	ess=[]
	out_size_=1
	for k in range(out_size_):
		avgv.append(0);
		rss.append(0);
		ess.append(0);
	for j in range(len(ds.X)):
		avgv=avgv+(ds.y[j]*y_std1+y_mean1)
	avgv=avgv/len(ds.X)
	if out==1:
		f=open("out_nn"+str(direct)+"_"+str(output_idx)+"_"+str(pc_learn_dataset)+"_"+str(nnsize)+".txt","wt")
		# save normalization
		f2=open("out_nn_norms_x_"+str(direct)+"_"+str(output_idx)+"_"+str(pc_learn_dataset)+"_"+str(nnsize)+".txt","wt")
		numpy.savetxt(f2,[x_mean1,x_std1],fmt="%g",delimiter=" ")
		f2.close()
		f2=open("out_nn_norms_y_"+str(direct)+"_"+str(output_idx)+"_"+str(pc_learn_dataset)+"_"+str(nnsize)+".txt","wt")
		numpy.savetxt(f2,[y_mean1,y_std1],fmt="%g",delimiter=" ")
		f2.close()
	for j in range(len(ds.X)):
		ae=0
		for k in range(out_size_):
			ae=ae+math.fabs(y_std1[k]*(v[j][k]-ds.y[j][k]))
			ess[k]=ess[k]+math.fabs(y_std1[k]*(v[j][k]-ds.y[j][k]))*math.fabs(y_std1[k]*(v[j][k]-ds.y[j][k]))
			rss[k]=rss[k]+(y_std1[k]*v[j][k]-avgv[k])*(y_std1[k]*v[j][k]-avgv[k])
		e=e+ae
		good=1;
		for k in range(out_size_):
			if ds.y[j][k]==0.0:
				good=0
		if good==1:
			e1=0
			for k in range(out_size_):
				e1=e1+math.fabs(y_std1[k]*(v[j][k]-ds.y[j][k])/(ds.y[j][k]*y_std1[k]+y_mean1[k]))
			e1/=out_size_
			ne=ne+1
		else:
			e1=0
		ne=ne+1
		if e1>maxe:
			maxe=e1
		if ae>maxae:
			maxae=ae
		avge=avge+e1
		if out==1:
			st=""
			for k in range(len(YY[0])):
				st=st+" "+str(v[j][k]*y_std1[k]+y_mean1[k])+" "+str(YY[j][k])
			f.write(st+"\n")
	if out==1:
		f.close()
	if ne!=0:
		avge=avge/ne
	s=str(i)+" mine %2.3f sumsq %2.3f maxrel %2.2f avgrel %2.2f r2 " % (mine,e/(len(ds.X)*out_size_), maxe*100, avge*100)
	for k in range(out_size_):
		r2=rss[k]/(rss[k]+ess[k])
		s=s+"%2.2f "% r2
	print(s)
	return e/(len(ds.X)*out_size_)

if (len(sys.argv)<8):
	ds=DS()
	X = torch.tensor(ds.X, dtype=torch.float32)
	Y = torch.tensor(ds.y, dtype=torch.float32)
	dst = TensorDataset(X, Y)
	loader = DataLoader(dst, batch_size=32, shuffle=True)
	print("matrix formed")
	model = MLP(in_size, 1)
	criterion = nn.MSELoss()
	#criterion = nn.SmoothL1Loss(beta=1.0)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	n_epochs = niters
	best_loss = float("inf")
	print("training initiated")
	for epoch in range(n_epochs):
		model.train()
		epoch_loss = 0.0
		for xb, yb in loader:
			optimizer.zero_grad()
			pred = model(xb)
			loss = criterion(pred, yb)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item() * len(xb)
		epoch_loss /= len(dst)
		if epoch_loss < best_loss:
			best_loss = epoch_loss
			torch.save(model.state_dict(), "mlp_torch"+str(direct)+"_"+str(output_idx)+"_"+str(pc_learn_dataset)+"_"+str(nnsize)+".pt")
		model.eval()
		with torch.no_grad():
			v = model(X).numpy()
		test(v, ds, epoch, 0, best_loss)
	model.load_state_dict(torch.load("mlp_torch"+str(direct)+"_"+str(output_idx)+"_"+str(pc_learn_dataset)+"_"+str(nnsize)+".pt"))
else:
	model = MLP(in_size, 1)
	model.load_state_dict(torch.load(mlp_file))
print("testing")
ds=DS()
X = torch.tensor(ds.X, dtype=torch.float32)
Y = torch.tensor(ds.y, dtype=torch.float32)
model.eval()
with torch.no_grad():
	v = model(X).numpy()
test(v,ds,0,1,0)
print("testing dataset")
n=0
sets_list=[]
YY=[]
for i in dataset.keys():
	if (n>len(dataset)*pc_learn_dataset/100.0):
		sets_list.append(n)
	n=n+1
if len(sets_list)!=0:
	ds=DS()
	X = torch.tensor(ds.X, dtype=torch.float32)
	Y = torch.tensor(ds.y, dtype=torch.float32)
	model.eval()
	with torch.no_grad():
		v = model(X).numpy()
	test(v,ds,0,1,0)
