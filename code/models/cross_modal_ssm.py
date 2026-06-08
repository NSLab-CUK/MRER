import warnings
from dataclasses import dataclass
from typing import List, Optional

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

from mamba_ssm import Mamba
import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from models.mamba.bimamba import Mamba as BiMamba 
from models.mamba.mm_bimamba import Mamba as MMBiMamba 
from models.base import BaseNet


class CrossModalMambaLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        activation='Swish',
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        if activation == 'Swish':
            activation = Swish
        elif activation == "GELU":
            activation = torch.nn.GELU
        else:
            activation = Swish

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = MMBiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

        self.primary_downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(d_model),
        )

    def forward(
        self,
        primary_x, secondary_x, 
        primary_inference_params=None,
        secondary_inference_params=None
    ):
        primary_out1, secondary_out1 = self.mamba(primary_x, secondary_x, primary_inference_params, secondary_inference_params)
        primary_out = primary_x + self.norm1(primary_out1)
        secondary_out = secondary_x + self.norm2(secondary_out1)

        return primary_out, secondary_out


class CrossModalCNNLayer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.primary_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.primary_bn = nn.BatchNorm1d(output_size)

        self.secondary_conv = nn.Conv1d(input_size, output_size, 3, padding=1, dilation=dilation, bias=False)
        self.secondary_bn = nn.BatchNorm1d(output_size)

        self.relu = nn.ReLU()

        self.primary_drop = nn.Dropout(dropout)
        self.secondary_drop = nn.Dropout(dropout)

        self.primary_net = nn.Sequential(self.primary_conv, self.primary_bn, self.relu, self.primary_drop)
        self.secondary_net = nn.Sequential(self.secondary_conv, self.secondary_bn, self.relu, self.secondary_drop)

        if input_size != output_size:
            self.primary_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
            self.secondary_skipconv = nn.Conv1d(input_size, output_size, 1, padding=0, dilation=dilation, bias=False)
        else:
            self.primary_skipconv = None
            self.secondary_skipconv = None

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.primary_conv.weight.data)
        nn.init.xavier_uniform_(self.secondary_conv.weight.data)

    def forward(self, x1, x2):
        primary_out = self.primary_net(x1)
        secondary_out = self.secondary_net(x2)
        if self.primary_skipconv is not None:
            x1 = self.primary_skipconv(x1)
        if self.secondary_skipconv is not None:
            x2 = self.secondary_skipconv(x2)
        primary_out = primary_out + x1
        secondary_out = secondary_out + x2
        return primary_out, secondary_out


class CrossModalSSM(nn.Module):
    def __init__(
        self,
        num_layers,
        input_size,
        output_sizes=[256, 512, 512],
        d_ffn=1024,
        activation='Swish',
        dropout=0.0,
        kernel_size=3,
        causal=False,
        mamba_config=None
    ):
        super().__init__()

        cnn_list = []
        mamba_list = []
        for i in range(len(output_sizes)):
            cnn_list.append(CrossModalCNNLayer(
                input_size=input_size if i < 1 else output_sizes[i-1],
                output_size=output_sizes[i],
                dropout=dropout
            ))
            mamba_list.append(CrossModalMambaLayer(
                d_model=output_sizes[i],
                d_ffn=d_ffn,
                dropout=dropout,
                activation=activation,
                causal=causal,
                mamba_config=mamba_config,
            ))

        self.mamba_layers = torch.nn.ModuleList(mamba_list)
        self.cnn_layers = torch.nn.ModuleList(cnn_list)

    def forward(
        self,
        primary_x, secondary_x,
        primary_inference_params=None,
        secondary_inference_params=None
    ):
        primary_out = primary_x
        secondary_out = secondary_x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            primary_out, secondary_out = cnn_layer(primary_out.permute(0, 2, 1), secondary_out.permute(0, 2, 1))
            primary_out = primary_out.permute(0, 2, 1)
            secondary_out = secondary_out.permute(0, 2, 1)
            primary_out, secondary_out = mamba_layer(
                primary_out, secondary_out,
                primary_inference_params=primary_inference_params,
                secondary_inference_params=secondary_inference_params
            )

        return primary_out, secondary_out
