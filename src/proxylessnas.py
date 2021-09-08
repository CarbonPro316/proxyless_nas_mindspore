import os
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P

import sys

# dj , remove it in PyTorch
# import torch.nn.init as init

from inspect import isfunction
from collections import OrderedDict


def drop_path(x, drop_prob):
    if drop_prob > 0.0001:
        keep_prob = 1. - drop_prob

        keep_prob = mindspore.Tensor(keep_prob, mindspore.float32)
        shape = x.shape

        dropout_gen_mask = mindspore.ops.DropoutGenMask()
        dropout_do_mask = mindspore.ops.DropoutDoMask()

        mask = dropout_gen_mask(shape, keep_prob)
        x = dropout_do_mask(x, mask, keep_prob)
    return x


def get_activation_layer(activation):
    assert (activation is not None)
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "relu6":
            return nn.ReLU6()
        else:
            raise NotImplementedError()
    else:
        assert (isinstance(activation, nn.Cell))
        return activation


class ConvBlock(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU())):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and (len(padding) == 4))

        if self.use_pad:
            self.pad = nn.ZeroPad2d(padding=padding)
            padding = 0
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, pad_mode='pad', padding=padding, dilation=dilation, group=groups,
                              has_bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps, momentum=0.9)
        if self.activate:
            self.activ = get_activation_layer(activation)

    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU())):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU())):
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class ProxylessBlock(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bn_eps,
                 expansion):
        super(ProxylessBlock, self).__init__()
        self.use_bc = (expansion > 1)
        mid_channels = in_channels * expansion

        if self.use_bc:
            self.bc_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_eps=bn_eps,
                activation="relu6")

        padding = (kernel_size - 1) // 2
        self.dw_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=mid_channels,
            bn_eps=bn_eps,
            activation="relu6")
        self.pw_conv = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=None)

    def construct(self, x):
        if self.use_bc:
            x = self.bc_conv(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ProxylessUnit(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bn_eps,
                 expansion,
                 residual,
                 shortcut):
        super(ProxylessUnit, self).__init__()
        assert (residual or shortcut)
        self.residual = residual
        self.shortcut = shortcut

        if self.residual:
            self.body = ProxylessBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bn_eps=bn_eps,
                expansion=expansion)

    def construct(self, x):
        if not self.residual:
            return x
        if not self.shortcut:
            return self.body(x)
        identity = x
        x = self.body(x)
        x = identity + x
        return x


class ProxylessNAS(nn.Cell):
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 residuals,
                 shortcuts,
                 kernel_sizes,
                 expansions,
                 bn_eps=1e-3,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(ProxylessNAS, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        init_block = conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            bn_eps=bn_eps,
            activation="relu6")

        # stage 1
        unit1 = ProxylessUnit(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=1,
            residual=True,
            shortcut=False)

        stage1 = OrderedDict([('unit1', unit1)])

        stage1 = nn.SequentialCell(stage1)

        # stage 2
        unit1 = ProxylessUnit(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=2,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=False)

        unit2 = ProxylessUnit(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit3 = ProxylessUnit(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=False,
            shortcut=True)

        unit4 = ProxylessUnit(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=False,
            shortcut=True)

        stage2 = OrderedDict([('unit1', unit1),
                              ('unit2', unit2),
                              ('unit3', unit3),
                              ('unit4', unit4)
                              ])

        stage2 = nn.SequentialCell(stage2)

        # stage 3
        unit1 = ProxylessUnit(
            in_channels=32,
            out_channels=40,
            kernel_size=7,
            stride=2,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=False)

        unit2 = ProxylessUnit(
            in_channels=40,
            out_channels=40,
            kernel_size=3,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit3 = ProxylessUnit(
            in_channels=40,
            out_channels=40,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit4 = ProxylessUnit(
            in_channels=40,
            out_channels=40,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        stage3 = OrderedDict([('unit1', unit1),
                              ('unit2', unit2),
                              ('unit3', unit3),
                              ('unit4', unit4)
                              ])

        stage3 = nn.SequentialCell(stage3)

        # stage 4
        unit1 = ProxylessUnit(
            in_channels=40,
            out_channels=80,
            kernel_size=7,
            stride=2,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=False)

        unit2 = ProxylessUnit(
            in_channels=80,
            out_channels=80,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit3 = ProxylessUnit(
            in_channels=80,
            out_channels=80,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit4 = ProxylessUnit(
            in_channels=80,
            out_channels=80,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit5 = ProxylessUnit(
            in_channels=80,
            out_channels=96,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=False)

        unit6 = ProxylessUnit(
            in_channels=96,
            out_channels=96,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit7 = ProxylessUnit(
            in_channels=96,
            out_channels=96,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit8 = ProxylessUnit(
            in_channels=96,
            out_channels=96,
            kernel_size=5,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        stage4 = OrderedDict([('unit1', unit1),
                              ('unit2', unit2),
                              ('unit3', unit3),
                              ('unit4', unit4),
                              ('unit5', unit5),
                              ('unit6', unit6),
                              ('unit7', unit7),
                              ('unit8', unit8),
                              ])

        stage4 = nn.SequentialCell(stage4)

        # stage 5
        unit1 = ProxylessUnit(
            in_channels=96,
            out_channels=192,
            kernel_size=7,
            stride=2,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=False)

        unit2 = ProxylessUnit(
            in_channels=192,
            out_channels=192,
            kernel_size=7,
            stride=1,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=True)

        unit3 = ProxylessUnit(
            in_channels=192,
            out_channels=192,
            kernel_size=7,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit4 = ProxylessUnit(
            in_channels=192,
            out_channels=192,
            kernel_size=7,
            stride=1,
            bn_eps=0.001,
            expansion=3,
            residual=True,
            shortcut=True)

        unit5 = ProxylessUnit(
            in_channels=192,
            out_channels=320,
            kernel_size=7,
            stride=1,
            bn_eps=0.001,
            expansion=6,
            residual=True,
            shortcut=False)

        stage5 = OrderedDict([('unit1', unit1),
                              ('unit2', unit2),
                              ('unit3', unit3),
                              ('unit4', unit4),
                              ('unit5', unit5),
                              ])

        stage5 = nn.SequentialCell(stage5)

        final_block = conv1x1_block(
            in_channels=320,
            out_channels=1280,
            bn_eps=bn_eps,
            activation="relu6")

        final_pool = nn.AvgPool2d(kernel_size=7, stride=1, pad_mode='valid')

        # combine from init_block to final_pool

        features = OrderedDict([('init_block', init_block),
                                ('stage1', stage1),
                                ('stage2', stage2),
                                ('stage3', stage3),
                                ('stage4', stage4),
                                ('stage5', stage5),
                                ('final_block', final_block),
                                ('final_pool', final_pool),
                                ])
        self.features = nn.SequentialCell(features)

        # output layer

        in_channels = final_block_channels

        self.output = nn.Dense(in_channels=in_channels, out_channels=num_classes)

        # dj , add it
        self.drop_prob = 0.0
        #######################

    def construct(self, x):
        x = self.features(x)

        # dj , add it
        x = drop_path(x, self.drop_prob)
        ######################

        x = P.Reshape()(x, (P.Shape()(x)[0], -1,))
        x = self.output(x)
        return x


def proxylessnas_gpu():
   residuals = [[1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]
   channels = [[24], [32, 32, 32, 32], [56, 56, 56, 56], [112, 112, 112, 112, 128, 128, 128, 128],
               [256, 256, 256, 256, 432]]
   kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 3, 3], [7, 5, 5, 5, 5, 3, 3, 5], [7, 7, 7, 5, 7]]
   expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 6, 6, 6]]
   init_block_channels = 40
   final_block_channels = 1728

   shortcuts = [[0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0]]

   net = ProxylessNAS(
       channels=channels,
       init_block_channels=init_block_channels,
       final_block_channels=final_block_channels,
       residuals=residuals,
       shortcuts=shortcuts,
       kernel_sizes=kernel_sizes,
       expansions=expansions)

   return net

def proxylessnas_mobile(num_classes=1000):
    residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    channels = [[16], [32, 32, 32, 32], [40, 40, 40, 40], [80, 80, 80, 80, 96, 96, 96, 96],
                [192, 192, 192, 192, 320]]
    kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
    expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
    init_block_channels = 32
    final_block_channels = 1280

    shortcuts = [[0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0]]

    net = ProxylessNAS(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        residuals=residuals,
        shortcuts=shortcuts,
        kernel_sizes=kernel_sizes,
        expansions=expansions,
        num_classes=num_classes
    )

    return net


def _test():
    net = proxylessnas_mobile()

    # net.train()
    # net.eval()
    net.set_train(False)

    print('The net is')
    print(net)

    # sys.exit(0)

    # net.load_state_dict(torch.load('proxylessnas_gpu-0745-acca5941.pth'))
    # ckpt = mindspore.load_checkpoint('proxylessnas_gpu-0745-acca5941.ckpt')
    # mindspore.load_param_into_net(net, ckpt)

    randn = mindspore.ops.StandardNormal()
    x_dim = (1, 3, 224, 224)
    x = randn(x_dim)

    y = net(x)

    print('The x.shape is', x.shape)
    # print('The output y is')
    # print(y)
    print('The y.shape is', y.shape)


if __name__ == "__main__":
    _test()
