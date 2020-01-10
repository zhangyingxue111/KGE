import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable


CUDA = torch.cuda.is_available()

 
class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        # self.non_linearity=nn.RReLU(0.1,0.3)
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)

class MLP(nn.Module):
    def __init__(self,input_size,output_size):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_size,output_size)
    def forward(self,x):
        x = F.tanh(self.fc1(x))
        return x
                                    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):        #in_features, out_features:nhid * nheads,nheads * nhid,
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim
        # --------------------------------修改
        # self.a = nn.Parameter(torch.zeros(
        #     size=(out_features, 2 * in_features + nrela_dim)))
        self.a__2 = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        nn.init.xavier_normal_(self.a__2.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))     #D*300
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))     #1*D
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.prelu=nn.PReLU()
        self.rrelu=nn.RReLU(0.1,0.3)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, edge_embed, edge_list_nhop, edge_embed_nhop):     #(x, edge_list, edge_embed,edge_list_nhop, edge_embed_nhop)
        N = input.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)   #头节点尾节点 邻居
        edge_embed = torch.cat(
            (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)     #边 邻居 
        # print("edge_embed:"+str(edge_embed.size()))

        edge_h = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E
        # print("edge_h:"+str(edge_h.size()))
        # edge_h:torch.Size([300, 285337])

        
        # ---------------------------------------修改:edge_h: (1*in_dim + nrela_dim) x E
        # edge_h = torch.cat(
        #     (input[edge[0, :], :], edge_embed[:, :]), dim=1).t()
        edge_h_2 = torch.cat(
            (input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim=1).t()

        
        edge_m = self.a.mm(edge_h)                  #式五
        # print("edge_m:"+str(edge_m.size()))
        # edge_m:torch.Size([100, 285337]) 

        # edge_m: D * E
        edge_m_2=self.a__2.mm(edge_h_2)


        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())     #式六:b(i,j,k)====1*D D*E=1*E
        
        # # -----------------------------------------------------------------------添加r层的权重 b'=1/|r|<sum>b(i,k)*tanh(W*c(ijk)+b)
        # print("powers.size:"+str(powers.size()))     # powers.size:torch.Size([285337])
        # print("edge_m.size:"+str(edge_m.size()))     # edge_m.size:torch.Size([100, 285337])
        # print("edge_m.t.size:"+str(edge_m.t().size()))     # edge_m.t.size:torch.Size([285337,100])
        input_size=edge_m.size()[0]
        #output_size=edge_m.size()[0]
        model=MLP(input_size,1)
        if CUDA:
            model.cuda()
        y=model(edge_m.t())
        # y.size(285337,1)   E*1
        # w=nn.Parameter(torch.zeros(size=edge_m.size()))
        # y=y.mm(w)

        powers=torch.mul(powers,y.t())/edge_embed.size()[0]
        # print("powers.size:"+str(powers.size()))     # powers.size:torch.Size([1, 285337])
        powers = powers.view(powers.size()[1])
        # print("after_view:powers.size:"+str(powers.size()))     # powers.size:torch.Size([1, 285337])
        # ====================================================================================
        edge_e = torch.exp(powers).unsqueeze(1)      #exp(b(i,j,k))       E*1
        assert not torch.isnan(edge_e).any()         
        # edge_e: E 
        
        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        e_rowsum[e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        # edge_e: E
        # 修改：----------------------------edge_m----》edge_m_2
        edge_w = (edge_e * edge_m).t()
        # edge_w: E * D

        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)                       #式7
        # h_prime: N x out

        assert not torch.isnan(h_prime).any()
        if self.concat:
            # if this layer is not last layer,级联
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
