import torch
import torch.nn as nn
import torch.nn.functional as F

class rfb_net(nn.Module):
    def __init__(self, inc):
        super(rfb_net, self).__init__()
        self.one = nn.Sequential(nn.Conv2d(inc, inc*2, kernel_size=1, padding=0),
                                 nn.Conv2d(inc*2, inc, kernel_size=(3, 3), padding=(1, 1)))

        self.two = nn.Sequential(nn.Conv2d(inc, inc*2, kernel_size=1, padding=0),
                                    nn.Conv2d(inc*2, inc*2, kernel_size=(1, 3), padding=(0, 1)),
                                    nn.Conv2d(inc*2, inc*2, kernel_size=(1, 3), padding=(0, 1)),
                                    nn.Conv2d(inc*2, inc, kernel_size=(3, 3), padding=(2, 2), dilation=2))

        self.three = nn.Sequential(nn.Conv2d(inc, inc*2, kernel_size=1, padding=0),
                                   nn.Conv2d(inc * 2, inc * 2, kernel_size=(3, 1), padding=(1, 0)),
                                   nn.Conv2d(inc * 2, inc * 2, kernel_size=(3, 1), padding=(1, 0)),
                                   nn.Conv2d(inc * 2, inc, kernel_size=(3, 3), padding=(2, 2), dilation=2)
                                   )

        self.four = nn.Conv2d(inc, inc, kernel_size=1)

        self.concatconv = nn.Conv2d(inc, inc, kernel_size=1)

    def forward(self, x):
        feature = []
        feature.append(self.one(x))
        feature.append(self.two(x))
        feature.append(self.three(x))
        feature.append(self.four(x))
        temp = self.concatconv(feature[0] + feature[1] + feature[2])

        out = 0.5 * temp + feature[3]
        return out
'''
if __name__ == "__main__":
    # from torch.autograd import Variable
    x = torch.rand([1, 3, 256, 256])
    rfb = rfb_net()
    out = rfb(x)
    print(out.size())
'''

