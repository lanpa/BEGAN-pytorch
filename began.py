from tensorboard import SummaryWriter
import torch
from torch.autograd import Variable
import argparse
import model
import model128
from dataloader import ImageFolderSR
from dexter import *
from datetime import datetime
import torchvision.utils as vutils
import math
import numpy as np
import socket
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='128_crop',  help='path to dataset')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--nz', type=int, help='size of random vector', default=64)
parser.add_argument('--imsize', type=int, help='size of image', default=64)
parser.add_argument('--gamma', type=float, help='gamma', default=0.5)
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--comment', help='comments', default='')

opt = parser.parse_args()
print(opt)
opt.batchSize = opt.batchSize * torch.cuda.device_count()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal(m.weight)


dataset = ImageFolderSR(root=opt.dataroot, HRsize=opt.imsize, is_crop=False)#, transform=inputTF, target_transform=GTTF)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), pin_memory=True, drop_last=True)
assert dataset
netG = model.G(n=128, h=opt.nz).cuda()
netD = model.D(n=128, h=opt.nz).cuda()

if opt.imsize==128:
    netG = model128.G(n=128, h=opt.nz).cuda()
    netD = model128.D(n=128, h=opt.nz).cuda()

netG = makeParallel(netG)
netD = makeParallel(netD)
#netG.apply(weights_init)
#netD.apply(weights_init)


optimG = torch.optim.Adam(netG.parameters(), lr = opt.lr, betas=(0.5, 0.999))
optimD = torch.optim.Adam(netD.parameters(), lr = opt.lr, betas=(0.5, 0.999))
if opt.workers==0:
    expname = ''
else:
    expname = '-'.join(['b_'+str(opt.batchSize), 'nz_'+str(opt.nz), 'gm_'+str(opt.gamma)])
writer = SummaryWriter('runs/'+socket.gethostname()+'-'+datetime.now().strftime('%B%d-%H-%M-%S')+expname+opt.comment)

def L_Df(v):
    #target = v.clone().detach()
    reconstructed, _ = netD(v)
    return (reconstructed-v).abs().mean()

kt = 0
lamk = 0.001
fixedNoise = Variable(torch.randn(opt.batchSize, opt.nz,1,1)).cuda()

for epoch in range(1000):
    for i, data in enumerate(dataloader, 1):
        n_iter = i+epoch*len(dataloader)
        xR = Variable(data[0]).cuda()

        z = torch.randn(opt.batchSize, opt.nz, 1, 1)
        xG = netG(Variable(z).cuda())
        optimG.zero_grad()

        L_G = L_Df(xG)
        L_G.backward(retain_variables=True)

        optimG.step()
        optimD.zero_grad()
        d_real = L_Df(xR)
        d_fake = L_G#L_Df(xG)
        L_D = d_real-kt*d_fake
        L_D.backward()
        optimD.step()

        L_D_val = L_D.data[0]
        L_G_val = L_G.data[0]


        kt = kt+lamk*(opt.gamma*L_D_val-L_G_val)
        if kt<0:
            kt = 0
        M_global = L_D_val + math.fabs(opt.gamma*L_D_val-L_G_val)

        writer.add_scalar('misc/M_global', M_global, n_iter)
        writer.add_scalar('misc/kt', kt, n_iter)
        writer.add_scalar('loss/L_D', L_D_val, n_iter)
        writer.add_scalar('loss/L_G', L_G_val, n_iter)
        writer.add_scalar('loss/d_real', d_real.data[0], n_iter)
        writer.add_scalar('loss/d_fake', d_fake.data[0], n_iter)
        if n_iter%100==1:
            for name, param in netD.named_parameters():
                if param.grad:
                    writer.add_scalar('D_grad/'+name, param.grad.abs().mean().data[0], n_iter)
            for name, param in netG.named_parameters():
                if param.grad:
                    writer.add_scalar('G_grad/'+name, param.grad.abs().mean().data[0], n_iter)
        LD_LG = L_D_val-L_G_val
        log_variable(M_global, L_D_val, L_G_val, kt, LD_LG)
        if n_iter%10000==0:
            opt.lr = opt.lr/2
            for param_group in optimD.param_groups:
                param_group['lr'] = opt.lr#param_group['lr']/2
            for param_group in optimG.param_groups:
                param_group['lr'] = opt.lr#param_group['lr']/2
            
        if n_iter%1000==1:
            writer.add_scalar('misc/learning', opt.lr, n_iter)
            print('dumping histogram')
            xG = netG(Variable(z).cuda())
            reconstructed, z = netD(xR)
            x = torch.cat([vutils.make_grid(reconstructed.data.cpu()/2+0.5, normalize=True, scale_each=True), vutils.make_grid(reconstructed.data.cpu()/2+0.5, normalize=False, scale_each=False)], 2)
            writer.add_image('reconstructed real image', x, n_iter)
            reconstructed, z = netD(xG)            
            x = torch.cat([vutils.make_grid(reconstructed.data.cpu()/2+0.5, normalize=True, scale_each=True), vutils.make_grid(reconstructed.data.cpu()/2+0.5, normalize=False, scale_each=False)], 2)
            writer.add_image('reconstructed fake image', x, n_iter)
            x = torch.cat([vutils.make_grid(xG.data.cpu()/2+0.5, normalize=True, scale_each=True), vutils.make_grid(xG.data.cpu()/2+0.5, normalize=False, scale_each=False)], 2)
            writer.add_image('generated fake image', x, n_iter)
            x = torch.cat([vutils.make_grid(netG(fixedNoise).data.cpu()/2+0.5, normalize=True, scale_each=True), vutils.make_grid(netG(fixedNoise).data.cpu()/2+0.5, normalize=False, scale_each=False)], 2)
            writer.add_image('generated fake image with fixed noise', x, n_iter)
            torch.save(netD, 'netD'+socket.gethostname()+'.pth')
            torch.save(netG, 'netG'+socket.gethostname()+'.pth')            
            for name, param in netG.named_parameters():
                if 'bn' in name:
                    continue
                writer.add_histogram('weight_G/'+name, param.clone().cpu().data.numpy(), n_iter)
                writer.add_histogram('grad_G/'+name, param.grad.clone().cpu().data.numpy(), n_iter)
                
            for name, param in netD.named_parameters():                
                if 'bn' in name:
                    continue
                writer.add_histogram('weight_D/'+name, param.clone().cpu().data.numpy(), n_iter)
                writer.add_histogram('grad_D/'+name, param.grad.clone().cpu().data.numpy(), n_iter)
