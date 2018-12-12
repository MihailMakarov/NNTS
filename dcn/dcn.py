import torch.nn as nn

class DCN(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, dilations: list, res_channels: int, activator_module=nn.Tanh()):
        super().__init__()
        layer_list=[nn.Conv1d(in_channels, res_channels, kernel_size=1, dilation=1)]

        for d in dilations:
            layer_list.append(nn.Conv1d(res_channels, res_channels, kernel_size=2, dilation=d))
            layer_list.append(activator_module)

        layer_list.append(nn.Conv1d(res_channels, out_channels, kernel_size=1, dilation=1))

        self.model = nn.Sequential(*layer_list)


    def receptive_fields_count(self):
        convolutions = list(filter(lambda x: type(x) is nn.Conv1d, list(self.model.modules())))
        dilations = [conv.dilation[0] for conv in convolutions if conv.kernel_size[0] == 2]
        return sum(dilations)+1

    def forward(self,data):
        result = self.model(data)
        return result

