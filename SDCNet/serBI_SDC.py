import torch 
import sys
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class BI_SDC(nn.Module):
	def __init__(self, args, field=None):
		super(BI_SDC, self).__init__()
		self.args = args

		V = args.embed_num
		D = args.embed_dim#300
		C = args.class_num
		N = args.num_layers
		H = args.hidden_size#100
		Ci = 1
		Co = args.kernel_num
		Ks = args.kernel_sizes

		self.embed_A = nn.Embedding(V, D)
		self.embed_B = nn.Embedding(V, D)
		self.embed_B.weight.data.copy_(field.vocab.vectors)

		self.conv_embed = nn.Conv2d(2, 1, (1, 1))
		self.conv_embed1 = nn.Conv2d(2, 1, (1, 2*H))

		self.tanh1 = nn.Tanh()
		self.w1 = nn.Parameter(torch.zeros(D))

		self.lstm = nn.LSTM(D, H, num_layers=N, \
							bidirectional = True,
							batch_first = True,
							dropout=args.LSTM_dropout)
		self.tanh2 = nn.Tanh()
		self.w2 = nn.Parameter(torch.zeros(H*2))
		self.conv1_D = nn.ModuleList(
						[nn.Conv2d(Ci, Co, (1, 2*H)) for _ in range(4)])
		self.convK_1 = nn.ModuleList(
						[nn.Conv2d(3*Co, Co, (K, 1), padding=(i,0)) \
						for i, K in enumerate(Ks)])
		self.conv3 = nn.Conv2d(4*Co, Co, (3, 1), padding=(1,0))
		self.conv4 = nn.Conv2d(5*Co, 2*Co, (3, 1), padding=(1,0))
		self.CNN_dropout = nn.Dropout(args.CNN_dropout)
		self.fc1 = nn.Linear(2*len(Ks)*Co, C)


		

	def forward(self, x):

		x_A = self.embed_A(x)  		# x [batch_size, sen_len, D]
		x_B = self.embed_B(x)
#		x = torch.cat([x_A.unsqueeze(3), x_B.unsqueeze(3)], 3)
#		x = x.permute(0,3,1,2)
#		x_embed = self.conv_embed(x)#[64 1 53 300]
	#	print(x_embed.size())
	#	input()
#		x = x_embed.squeeze(1)#[64 53 300]
	#	print(x.size())
		x=torch.add(x_A,x_B)
		W = self.tanh1(x)
		W_alpha = F.softmax(torch.matmul(W, self.w1), dim=1).unsqueeze(-1)
		x = W * W_alpha
		

		H, _  = self.lstm(x)	# H [batch_size, sen_len, 2*H]
		S = self.tanh2(H)
		S_alpha = F.softmax(torch.matmul(S, self.w2), dim=1).unsqueeze(-1)
		out = H * S_alpha		# out [batch_size, sen_len, 2*H][64 56 200]
		#print(out.size())
		#input()
		x_embed = out.unsqueeze(1)	# x_ [batch_size, 1, sen_len, 2*H]
	#	x_embed=x_embed.permute(0,3,1,2)
		#x_embed = self.conv_embed1(x_embed)
		xD_out = [conv(x_embed) for conv in self.conv1_D]#[64,100,45,1]
		#print(xD_out[1].size())
		#input()						# xD_out[i] [batch_size, Co, sen_len, 1]
		xK_in = [torch.cat([x_embed.permute(0,3,2,1), d], 1) for d in xD_out]#[64,300,45,1]
		#print(xK_in[1].size())
		#input()						# x[i] [batch_size, Co+2*H, sen_len, 1]	

		xK_out = [F.relu(conv(xK_in[i])) for i, conv in enumerate(self.convK_1)]

		x3_in = [torch.cat([x_embed.permute(0,3,2,1), d], 1) for d in xD_out]
		x3_in = [torch.cat([x3_in[i], d], 1) for i, d in enumerate(xK_out)]

		x3_out = [F.relu(self.conv3(i)) for i in x3_in]

		x4_in = [torch.cat([x_embed.permute(0,3,2,1), d], 1) for d in xD_out]
		x4_in = [torch.cat([x4_in[i], d], 1) for i, d in enumerate(xK_out)]
		x4_in = [torch.cat([x4_in[i], d], 1) for i, d in enumerate(x3_out)]
#		print(x4_in[1].size())
#		input()
		x4 = [F.relu(self.conv4(i)) for i in x4_in]
		x = [torch.add(i.squeeze(3), x_embed.squeeze(1).permute(0,2,1)) \
			 for i in x4]#short-cut
		x = torch.cat(x, 1)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
	#	x = torch.cat([x, out[:,-1,:]], 1)
		x = self.CNN_dropout(x)
		logit = self.fc1(x)
		
		return logit
