"""

DRQN-based agent that learns to communicate with other agents to play
the Switch game.

"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from pysc2.lib import features
import numpy as np
from sc2.flat_features import FLAT_FEATURES
from sc2.convgru import ConvGRU

NUM_FUNCTIONS = 2
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


class Sc2CNet(nn.Module):

    def __init__(self, opt):
        super(Sc2CNet, self).__init__()

        self.opt = opt
        self.comm_size = opt.game_comm_bits
        self.init_param_range = (-0.08, 0.08)
        self.data_format = 'NCHW'
        self.ch = opt.static_shape_channels
        self.res = opt.resolution
        self.size2d = [opt.resolution, opt.resolution]
        self.screen_embed_spatial_conv = {}
        self.create_screen_embed_obs()

        self.screen_input_conv = nn.Sequential(
            nn.Conv2d(in_channels=52, out_channels=16,
                      kernel_size=7, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # self.fn_conv = nn.Conv2d(32, num_units, kernel_size=1, stride=1)
        self.convgru = ConvGRU(input_size=(24, 24),
                    input_dim=34,
                    hidden_dim=[34,34],
                    kernel_size=(3,3),
                    num_layers=2,
                    dtype=torch.FloatTensor,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=19584, out_features=256),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        self.fn_non_spatial_output = nn.Sequential(
            nn.Linear(256, NUM_FUNCTIONS+opt.game_comm_bits),
            nn.Softmax()
        )
        self.world_output = nn.Sequential(
            nn.Conv2d(34, 1, kernel_size=1, stride=1),
            nn.Flatten(),
            nn.Linear(opt.resolution * opt.resolution,
                      opt.resolution*opt.resolution)
        )

    def get_params(self):
        return list(self.parameters())

    def create_screen_embed_obs(self):
        for s in features.SCREEN_FEATURES:
            if s.type == features.FeatureType.CATEGORICAL:
                dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                self.screen_embed_spatial_conv[s.index] = nn.Sequential(
                    nn.Conv2d(
                        in_channels=s.scale,
                        out_channels=dims,
                        kernel_size=1,
                        stride=1
                    ),
                    nn.BatchNorm2d(dims),
                    nn.ReLU()
                )
                self.add_module(
                    f"screen_embed_spatial_conv:{s.name}", self.screen_embed_spatial_conv[s.index])

    def screen_embed_obs(self, x):
        print(x.shape)
        feats = list(x.split(1, dim=-1))
        out_list = []
        for s in features.SCREEN_FEATURES:
            f = feats[s.index]
            if s.type == features.FeatureType.CATEGORICAL:
                f = torch.squeeze(f, -1).type(torch.LongTensor)
                indices = torch.nn.functional.one_hot(f, num_classes=s.scale)
                x = self.from_nhwc(indices.type(torch.FloatTensor))
                out = self.screen_embed_spatial_conv[s.index](x)
                out = self.to_nhwc(out)
            elif s.type == features.FeatureType.SCALAR:
                out = self.log_transform(
                    f.type(torch.FloatTensor), s.scale)
            out_list.append(out)
        return torch.cat(out_list, dim=-1).to(device)

    def log_transform(self, x, scale):
        return torch.log(x + 1.)

    def flat_embed_obs(self, x):
        spec = FLAT_FEATURES
        feats = list(x.split(1, dim=-1))
        out_list = []
        for s in spec:
            f = feats[s.index]
            if s.type == features.FeatureType.CATEGORICAL:
                dims = np.round(np.log2(s.scale)).astype(np.int32).item()
                dims = max(dims, 1)
                indices = torch.nn.functional.one_hot(
                    torch.squeeze(f, -1), s.scale)
                out = self.embed_flat_fc[s.index](indices)
            elif s.type == features.FeatureType.SCALAR:
                out = self.log_transform(f.type(torch.FloatTensor), s.scale)
            out_list.append(out)
        return torch.cat(out_list, dim=-1).to(device)

    def concat2d(self, lst):
        if self.data_format == 'NCHW':
            return torch.cat(lst, dim=1).to(device)
        return torch.cat(lst, dim=3).to(device)

    def broadcast_along_channels(self, flat, size2d):
        if self.data_format == 'NCHW':
            return flat.unsqueeze(2).unsqueeze(3).repeat(1, 1, size2d[0], size2d[1])
        return flat.unsqueeze(1).unsqueeze(2).repeat(1, size2d[0], size2d[1], 1)

    def to_nhwc(self, map2d):
        if self.data_format == 'NCHW':
            return map2d.permute(0, 2, 3, 1)
        return map2d

    def from_nhwc(self, map2d):
        if self.data_format == 'NCHW':
            return map2d.permute(0, 3, 1, 2)
        return map2d

    def reset_parameters(self):
        opt = self.opt
        # self.messages_mlp.linear1.reset_parameters()
        # self.rnn.reset_parameters()
        # self.agent_lookup.reset_parameters()
        # self.state_lookup.reset_parameters()
        # self.prev_action_lookup.reset_parameters()
        # if self.prev_message_lookup:
        #     self.prev_message_lookup.reset_parameters()
        # if opt.comm_enabled and opt.model_dial:
        #     self.messages_mlp.batchnorm1.reset_parameters()
        # self.outputs.linear1.reset_parameters()
        # self.outputs.linear2.reset_parameters()
        # for p in self.rnn.parameters():
        #     p.data.uniform_(*self.init_param_range)

    def forward(self, s_t, messages, hidden, prev_action, agent_index):
        opt = self.opt

        # s_t = Variable(s_t)
        # hidden = Variable(hidden)
        # prev_message = None
        # if opt.model_dial:
        # 	if opt.model_action_aware:
        # 		prev_action = Variable(prev_action)
        # else:
        # 	if opt.model_action_aware:
        # 		prev_action, prev_message = prev_action
        # 		prev_action = Variable(prev_action)
        # 		prev_message = Variable(prev_message)
        # 	messages = Variable(messages)
        # agent_index = Variable(agent_index)

        # z_a, z_o, z_u, z_m = [0]*4
        # z_a = self.agent_lookup(agent_index)
        # z_o = self.state_lookup(s_t)
        # if opt.model_action_aware:
        # 	z_u = self.prev_action_lookup(prev_action)
        # 	if prev_message is not None:
        # 		z_u += self.prev_message_lookup(prev_message)
        # z_m = self.messages_mlp(messages.view(-1, self.comm_size))

        # z = z_a + z_o + z_u + z_m
        # z = z.unsqueeze(1)

        # rnn_out, h_out = self.rnn(z, hidden)
        # outputs = self.outputs(rnn_out[:, -1, :].squeeze())
        flat_input = messages.flatten(1)
        screen_input = self.to_nhwc(torch.stack([torch.from_numpy(x["feature_screen"]) for x in s_t]))
        screen_emb = self.screen_embed_obs(screen_input)
        flat_emb = self.flat_embed_obs(flat_input)
        # screen_emb = self.layer_norm(screen_emb)
        screen_out = self.screen_input_conv(self.from_nhwc(screen_emb))
        broadcast_out = self.broadcast_along_channels(flat_emb, self.size2d)
        state_out = self.concat2d([screen_out, broadcast_out])
        # state_out = screen_out
        state_out, h = self.convgru(state_out.unsqueeze(dim=1))
        state_out = state_out[0].squeeze(1)
        x = self.to_nhwc(state_out)
        fc = self.fc1(x)
        fn_out = self.fn_non_spatial_output(fc)

        world_out = self.world_output(state_out)

        policy = (fn_out, world_out)
        h_out = h[0][0]

        return h_out, policy
