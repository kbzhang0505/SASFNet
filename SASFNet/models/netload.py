import torch
import networks
import functools
import torch.nn as nn
'''
对pre进行修改，更新model_dict
'''
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

norm_layer = get_norm_layer(norm_type='instance')
# 实例化net
net = networks.NLayerDiscriminator(3, 64, n_layers=3, norm_layer=norm_layer, use_sigmoid=True,
                                   gpu_ids=[0], use_parallel=False)
# a = torch.randn([1,3,256,256])
# b = net(a)
# print(b.shape)
# 加载权重
pre = torch.load(r"/media/ubuntu/PortableSSD/RTX3090/code/Single_gan/checkpoints/soft_assis_cal/1360_net_D.pth")

for n, v in pre.items():
	torch.nn.init.xavier_uniform_(v)


for m in net.modules():
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)


# 权重字典 items是一个元组，2项，第一项keys 第二项parameter
# model_dict = net.state_dict()
'''
for i in model_dict.items():
    i = list(i)
    if i[0].find("soft_edge") != -1:
        print(i[0])
        for name, parameters in pre.items():
            if name == i[0]:
                i[1] = parameters
'''
# net.load_state_dict(model_dict)
# torch.save(net, r"/media/ubuntu/PortableSSD/RTX3090/code/Single_gan/checkpoints/soft_assis_coo/0_net_D.pth.pth")


# for name, parameters in pre.items():
#     print(name)
#
#     if name.find("Conv") != -1:
#         print(parameters.shape)

# print(type(model_dict))
# for i in model_dict.items():
#      print(i[0])
# model_dict.update(pre)
'''
state_dict = {k:v for k,v in pre.items() if k in model_dict.keys()}
for k,v in pre.items():
    #s = str(k).split('.')
    #if len(s) >= 3:
    #    if s[0]+'.'+s[1]+'.'+s[2] == "body.ec_RFEB1.1":
    if k[0:15] == "body.ec_RFEB1.2":
        state_dict[k] = pre[k[0:14]+'1'+k[15:]]
        print(k)
        #print(v)
'''
'''
model_dict.update(state_dict)
netccoa(model_dict)
ad('pretext.pth')
model2_dict = net.state_dict()
state_dict = {k:v for k,v in pretext_model.items() if k in model2_dict.keys()}
model2_dict.update(state_dict)
net.load_state_dict(model2_dict)

torch.save(net.cpu().state_dict(), "/home/amax/HJH/Single_Gan/checkpoints/soft_assis_cal/1150_net_G.pth")
#net.save("/home/amax/HJH/Single_Gan/checkpoints/soft_assis_cal/1150_net_G.pth")
'''
