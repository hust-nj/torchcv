import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.backbone_selector import BackboneSelector
from model.tools.module_helper import ModuleHelper
from model.seg.loss.loss import BASE_LOSS_DICT
from model.seg.ops.nonlocal_block_nowd import NonLocal2d_nowd


class _ConvBatchNormReluBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=1, dilation=1, norm_type=None):
        super(_ConvBatchNormReluBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=inplanes, out_channels=outplanes,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn_relu = ModuleHelper.BNReLU(outplanes, norm_type=norm_type)

    def forward(self, x):
        x = self.bn_relu(self.conv(x))
        return x


class NLModule_nowd(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, configer):
        super(NLModule_nowd, self).__init__()
        inter_channels = in_channels // 2
        self.configer = configer

        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, norm_type=self.configer.get('network', 'norm_type')))

        self.ctb = NonLocal2d_nowd(inter_channels, inter_channels // 2,
                                   downsample=self.configer.get('nonlocal', 'downsample'),
                                   whiten_type=self.configer.get('nonlocal', 'whiten_type'), 
                                   weight_init_scale=self.configer.get('nonlocal', 'weight_init_scale'),
                                   with_gc=self.configer.get('nonlocal', 'with_gc'),
                                   with_nl=self.configer.get('nonlocal', 'with_nl'),
                                   nowd=self.configer.get('nonlocal', 'nowd'),
                                   use_out=self.configer.get('nonlocal', 'use_out'),
                                   out_bn=self.configer.get('nonlocal', 'out_bn'))

        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   ModuleHelper.BNReLU(inter_channels, norm_type=self.configer.get('network', 'norm_type')))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=1, padding=1, dilation=1, bias=False),
            ModuleHelper.BNReLU(out_channels, norm_type=self.configer.get('network', 'norm_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        if self.ctb is not None:
            for i in range(recurrence):
                output = self.ctb(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class NonLocalNet_nowd(nn.Sequential):
    def __init__(self, configer):
        super(NonLocalNet_nowd, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()
        num_features = self.backbone.get_num_features()
        self.dsn = nn.Sequential(
            _ConvBatchNormReluBlock(160, 64, 3, 1,
                                    norm_type=self.configer.get('network', 'norm_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, self.num_classes, 1, 1, 0)
        )
        self.nlm = NLModule_nowd(num_features, 512, self.num_classes, self.configer)

        self.valid_loss_dict = configer.get('loss', 'loss_weights', configer.get('loss.loss_type'))

    def forward(self, data_dict):
        x = self.backbone(data_dict['img'])
        aux_x = self.dsn(x[-2])
        x = self.nlm(x[-1])
        x_dsn = F.interpolate(aux_x, size=(data_dict['img'].size(2), data_dict['img'].size(3)),
                              mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(data_dict['img'].size(2), data_dict['img'].size(3)),
                          mode="bilinear", align_corners=True)
        out_dict = dict(dsn_out=x_dsn, out=x)
        if self.configer.get('phase') == 'test':
            return out_dict

        loss_dict = dict()
        if 'dsn_ce_loss' in self.valid_loss_dict:
            loss_dict['dsn_ce_loss'] = dict(
                params=[x_dsn, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['dsn_ce_loss']])
            )

        if 'ce_loss' in self.valid_loss_dict:
            loss_dict['ce_loss'] = dict(
                params=[x, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ce_loss']])
            )

        if 'ohem_ce_loss' in self.valid_loss_dict:
            loss_dict['ohem_ce_loss'] = dict(
                params=[x, data_dict['labelmap']],
                type=torch.cuda.LongTensor([BASE_LOSS_DICT['ohem_ce_loss']]),
                weight=torch.cuda.FloatTensor([self.valid_loss_dict['ohem_ce_loss']])
            )
        return out_dict, loss_dict


if __name__ == '__main__':
    i = torch.Tensor(1, 3, 512, 512).cuda()
    model = PSPNet(num_classes=19).cuda()
    model.eval()
    o, _ = model(i)
    # print(o.size())
    # final_out = F.upsample(o,scale_factor=8)
    # print(final_out.size())
    print(o.size())
    print(_.size())
