import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import torch
import os
#加载tidigits数据集
from python_speech_features import fbank
import numpy as np
import scipy.io.wavfile as wav
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def read_data(path, n_bands, n_frames):
    overlap = 0.5

    # tidigits_file = 'data/tidigits/tidigits_{}_{}.pickle'.format(n_bands, n_frames)
    # if os.path.isfile(tidigits_file):
    #     print('Reading {}...'.format(tidigits_file))
    #     with open(tidigits_file, 'rb') as f:
    #         train_set, test_set = pickle.load(f)
    #     return train_set, test_set

    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.waV') and file[0] != 'O':
                filelist.append(os.path.join(root, file))
    filelist = filelist[:500]

    n_samples = len(filelist)

    def keyfunc(x):
        s = x.split('/')
        return (s[-1][0], s[-2], s[-1][1]) # BH/1A_endpt.wav: sort by '1', 'BH', 'A'
    filelist.sort(key=keyfunc)

    feats = np.empty((n_samples, n_bands * n_frames))
    labels = np.empty((n_samples,), dtype=np.int)
    with tqdm(total=len(filelist)) as pbar:
        for i, file in enumerate(filelist):
            pbar.update(1)
            label = file.split('\\')[-1][0]
            if label == 'Z':
                labels[i] = np.int(0)
            else:
                labels[i] = np.int(label)
            rate, sig = wav.read(file)
            duration = sig.size / rate
            winlen = duration / (n_frames * (1 - overlap) + overlap)
            winstep = winlen * (1 - overlap)
            feat, energy = fbank(sig, rate, winlen, winstep, nfilt=n_bands, nfft=4096, winfunc=np.hamming)
            # feat = np.log(feat)

            feats[i] = feat[:n_frames].flatten() # feat may have 41 or 42 frames

    feats = normalize(feats, norm='l2', axis=1)

    np.random.seed(42)
    p = np.random.permutation(n_samples)
    feats, labels = feats[p], labels[p]

    n_train_samples = int(n_samples * 0.7)
    print('n_train_samples:',n_train_samples)

    train_set = (feats[:n_train_samples], labels[:n_train_samples])
    test_set = (feats[n_train_samples:], labels[n_train_samples:])

    return train_set, train_set, test_set
# b= torch.tensor([[0.03,1,-0.5],[0.9,0.001,-1]])
# a=torch.nn.functional.softmax(b,dim=-2)
# print(a,'\n',torch.nn.functional.softmax(b,dim=-1),'\n',torch.nn.functional.softmax(b,dim=0),'\n',torch.nn.functional.softmax(b,dim=1))
a=torch.tensor([1.0,2.0,3.0])
print(a/a.max())
abc
n_bands, n_frames= 41, 40
train_loader, traintest_loader, test_loader = read_data(path='E:/学习/代偿运动理论/文章代码/SNU/DRTP4/DRTP_end/DATASETS/tidigits/isolated_digits_tidigits', n_bands=n_bands, n_frames=n_frames)#(2900, 1640) (2900,)

x = torch.tensor(train_loader[0]).float()
y = torch.tensor(train_loader[1]).long()
print(x)
y = torch.zeros(y.shape[0], 10).scatter_(1, y.unsqueeze(1), 1.0)
testx = torch.tensor(test_loader[0]).float()
testy = torch.tensor(test_loader[1]).long()
testy = torch.zeros(testy.shape[0], 10).scatter_(1, testy.unsqueeze(1), 1.0)
# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
#
# y = x ** 2 + 0.2 * torch.rand(x.size())
# # 得到x自乘的矩阵，然后加上同x矩阵相同的噪声
#
# print(x, y, x.shape,y.shape)

 
x,y = Variable(x),Variable(y)
#将矩阵转化为 变量
 
class Net(torch.nn.Module):
#定义神经网络
	def __init__(self,n_feature,n_hidden,n_output):
	#初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
		super(Net,self).__init__()
		#此步骤是官方要求
		self.hidden = torch.nn.Linear(n_feature,n_hidden)
		#设置输入层到隐藏层的函数
		self.predict = torch.nn.Linear(n_hidden,n_output)
		#设置隐藏层到输出层的函数

	def forward(self,x):
	#定义向前传播函数
		x = F.relu(self.hidden(x))
        #给x加权成为a，用激励函数将a变成特征b
		x = self.predict(x)
        #给b加权，预测最终结果
		return x
net = Net(n_frames*n_bands,500,10)

print(net)
#查看各层之间的参数

opt = torch.optim.Adam(net.parameters(), lr=0.01)
# 设置学习率为0.5，用随机梯度下降发优化神经网络的参数
a=[]
lossfunc = torch.nn.MSELoss()
# 设置损失函数为均方损失函数，用来计算每次的误差
l=100
for t in tqdm(range(l)):
    # 进行100次的优化
    prediction = net(x)
    # 得到预测值
    # if t == 9999:
    #     print('1',prediction)
    #     print('2',y)
    loss = lossfunc(prediction, y)
    if t == l-1:
        print('loss:',loss)
    a.append(loss)
    # print('loss:',t,loss)

    # 得到预测值与真实值之间的误差
    opt.zero_grad()
    # 梯度清零
    loss.backward()
    # 反向传播
    opt.step()
# 梯度优化
plt.plot(a)
plt.show()
plt.ioff()
prediction = net(testx)
# print(prediction.shape[0],testy.shape)
acc = 0
for i in range(prediction.shape[0]):
    q = testy[i].numpy().tolist()
    k = prediction[i].detach().numpy().tolist()
    # print(q,k)
    if k.index(max(k)) == q.index(1):
        acc+=1
print('acc:',acc/prediction.shape[0])