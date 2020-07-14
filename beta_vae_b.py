import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

from data import chairs_3d_iter, dsprites_iter

_use_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if _use_cuda else 'cpu')


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = std.new_tensor(torch.randn(std.size()))
    return mu + std * eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, input):
        return input.view(self.size)


# network borrow from https://github.com/1Konny/Beta-VAE/blob/master/model.py
class BetaVAE_B(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_B, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            View((-1, 32 * 4 * 4)),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, z_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 32 * 4 * 4),
            nn.ReLU(True),
            View((-1, 32, 4, 4)),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),
        )

    def forward(self, x):
        dist = self.encoder(x)
        mu = dist[:, :self.z_dim]
        logvar = dist[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_rec = self.decoder(z)

        return x_rec, mu, logvar


def train(model, data_iter, lr=5e-4, n_epochs=50, gamma=1000, min_C=1, max_C=30, save_dir='./beta_vae_b/'):
    os.makedirs(save_dir, exist_ok=True)
    trainer = optim.Adam(model.parameters(), lr, betas=[0.5, 0.999])
    c_it = 0
    total_iters = n_epochs * len(data_iter)
    c_linspace = torch.linspace(min_C, max_C, total_iters).to(DEVICE)
    for e in range(n_epochs):
        model.train()
        for b, x in enumerate(data_iter):
            x = x.to(DEVICE)
            rec_x, mu, logvar = model(x)
            kld_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1, 1))
            rec_loss = F.binary_cross_entropy_with_logits(rec_x, x, size_average=False) / rec_x.size(0)
            loss = rec_loss + gamma * torch.abs(kld_loss - c_linspace[c_it])
            c_it += 1

            model.zero_grad()
            loss.backward()
            trainer.step()

            if (b + 1) % 100 == 0:
                print('[ %2d / %2d ] [%5d] kld_loss: %.4f rec_loss: %4f' % (
                    e + 1, n_epochs, b + 1, kld_loss.item(), rec_loss.item()))

        # save
        if (e + 1) % 5 == 0:
            torch.save(model.state_dict(), save_dir + 'beta_vae_b_{}.pth'.format(e + 1))
        # test
        with torch.no_grad():
            # 随机生成
            model.eval()
            z = torch.randn(64, model.z_dim).to(DEVICE)
            test_ims = F.sigmoid(model.decoder(z))
            tv.utils.save_image(test_ims, save_dir + 'im_{}.png'.format(e + 1))

            # 对隐层进行线性插值
            for dim in range(model.z_dim):
                z = torch.randn(model.z_dim).repeat(8, 1)
                z[:, dim] = torch.linspace(-3, 3, 8)
                test_ims = F.sigmoid(model.decoder(z.to(DEVICE)))
                tv.utils.save_image(test_ims, save_dir + 'im_{}_d{}.png'.format(e + 1, dim))


if __name__ == '__main__':
    model2 = BetaVAE_B(nc=1).to(DEVICE)
    train(model2, data_iter=dsprites_iter, n_epochs=10, gamma=20, save_dir='./dsprites_b100/')

    # this is ok
    # model = BetaVAE_B(nc=3).to(DEVICE)
    # train(model, data_iter=chairs_3d_iter, n_epochs=50, gamma=100, save_dir='./3dchars_b100/')

    # this is ok
    model = BetaVAE_B(nc=3).to(DEVICE)
    train(model, data_iter=chairs_3d_iter, n_epochs=10, gamma=100, save_dir='./3dchars_n10/')

    # notes: 训练成功与否跟gamma的设置有关系，对于3dchairs数据集，若gamma设置为1000（原文设置），则训练失败，重构
    # 误差基本降不下去，对dsprites，若gamma设置为100（100在3dchairs数据及上ok）则失败，设置为20就ok。 原因？？？
