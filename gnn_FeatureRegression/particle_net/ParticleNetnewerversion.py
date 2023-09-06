''' ParticleNet

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570

Adapted from the DGCNN implementation in https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.
'''
import numpy as np
import torch
import torch.nn as nn


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)



class ParticleNetRegressorPFOnly(nn.Module):
    def __init__(self, input_dim=3):
        super(ParticleNetRegressorPFOnly, self).__init__()

        # Common ParticleNet module
        self.pn = ParticleNetFeatureExtractor(input_dim=input_dim)
        
        # Define dense networks for each particle pair
        self.met_1_net = self._make_dense_net(2)
        self.met_2_net = self._make_dense_net(2)
        self.met_3_net = self._make_dense_net(2)
        self.pair_1_2_net = self._make_dense_net(5)
        self.pair_1_3_net = self._make_dense_net(5)
        self.pair_2_3_net = self._make_dense_net(5)
        
    def _make_dense_net(self, output_size):
        return nn.Sequential(
            nn.Linear(self.pn.output_size, 64),  # Assuming the ParticleNet's output is named 'output_size'
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, pf_points, pf_features, pf_mask):
        common_features = self.pn(pf_points, pf_features, pf_mask)
        
        # Extracting outputs for each particle pair
        met_1_out = self.met_1_net(common_features)
        met_2_out = self.met_2_net(common_features)
        met_3_out = self.met_3_net(common_features)
        pair_1_2_out = self.pair_1_2_net(common_features)
        pair_1_3_out = self.pair_1_3_net(common_features)
        pair_2_3_out = self.pair_2_3_net(common_features)
        
        # Returning as a list
        return [met_1_out, met_2_out, met_3_out, pair_1_2_out, pair_1_3_out, pair_2_3_out]


class ParticleNetFeatureExtractor(nn.Module):
    def __init__(self, input_dim=3, use_fts_bn=True):
        super(ParticleNetFeatureExtractor, self).__init__()

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dim)

        # Simplify by using only two EdgeConv layers as an example
        self.edge_conv1 = EdgeConvBlock(k=16, in_feat=input_dim, out_feats=[64, 128])
        self.edge_conv2 = EdgeConvBlock(k=32, in_feat=128, out_feats=[128, 256])

        # Linear layers for feature extraction
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256)  # This is the feature vector we will extract
        )

    def forward(self, pf_points, pf_features, pf_mask):
        if self.use_fts_bn:
            pf_features = self.bn_fts(pf_features)
        
        x = self.edge_conv1(pf_points, pf_features, pf_mask)
        x = self.edge_conv2(pf_points, x, pf_mask)
        x = torch.mean(x, dim=2)  # Global average pooling
        features = self.fc(x)

        return features

        
        

class ParticleNetRegressor(nn.Module):
    def __init__(self, input_dims):
        super(ParticleNetRegressor, self).__init__()
        
        # Define the common ParticleNet features here
        # ... 

        # Define dense networks for each particle pair
        # Each dense network corresponds to one of the used_labels2 lists.
        self.particle_pair_nets = nn.ModuleDict({
            "MET_1": self._make_dense_net(2),
            "MET_2": self._make_dense_net(2),
            "MET_3": self._make_dense_net(2),
            "1_2": self._make_dense_net(5),
            "1_3": self._make_dense_net(5),
            "2_3": self._make_dense_net(5),
        })

    def _make_dense_net(self, output_size):
        # This can be further customized for deeper architectures if needed
        return nn.Sequential(
            nn.Linear(self.input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, pf_points, pf_features, pf_mask):
        common_features = self.pn(pf_points, pf_features, pf_mask)
        
        # Extracting outputs for each particle pair
        outputs = {pair: net(common_features) for pair, net in self.particle_pair_nets.items()}
        
        return outputs



class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 for_segmentation=False,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):
#         print('points:\n', points)
#         print('features:\n', features)
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output


class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)

class ParticleNetTaggerPFOnly(nn.Module):
    def __init__(self,
                 pf_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTaggerPFOnly, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf_points, pf_features, pf_mask):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask

        features = self.pf_conv(pf_features * pf_mask) * pf_mask
        return self.pn(pf_points, features, pf_mask)


class ParticleNetTagger(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask

        points = torch.cat((pf_points, sv_points), dim=2)
        features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
        mask = torch.cat((pf_mask, sv_mask), dim=2)
        return self.pn(points, features, mask)