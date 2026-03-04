import segmentation_models_pytorch as smp
import torch.nn as nn
import torch


class LandslideDeepLabV3(nn.Module):
    def __init__(
        self,
        in_channels=14,
        classes=2,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        mc_dropout=0.1,
    ):
        """
        Args:
            in_channels: 输入通道数 (默认 14)
            classes: 输出类别数 (默认 2)
            encoder_name: 骨干网络 (默认 resnet50)
            encoder_weights: 预训练权重 (默认 imagenet)
            mc_dropout: MC Dropout概率 (默认 0.1, 0表示禁用)
        """
        super(LandslideDeepLabV3, self).__init__()

        self.mc_dropout = mc_dropout

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

        if mc_dropout > 0:
            self._add_mc_dropout()

    def _add_mc_dropout(self):
        """在模型中插入MC Dropout层"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                module.p = self.mc_dropout

    def enable_mc_dropout(self, enable: bool = True):
        """启用/禁用MC Dropout模式"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                if enable:
                    module.train()
                else:
                    module.eval()

    def forward(self, x):
        return self.model(x)
