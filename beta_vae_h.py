import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv

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
class BetaVAE_H(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            View((-1, 256)),
            nn.Linear(256, z_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            View((-1, 256, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
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


def train(model, data_iter, lr=1e-3, n_epochs=10, beta=1, save_dir='./results/'):
    os.makedirs(save_dir, exist_ok=True)
    trainer = optim.Adam(model.parameters(), lr, betas=[0.5, 0.999])

    for e in range(n_epochs):
        model.train()
        for b, x in enumerate(data_iter):
            x = x.to(DEVICE)
            rec_x, mu, logvar = model(x)
            kld_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1, 1))
            rec_loss = F.binary_cross_entropy_with_logits(rec_x, x, size_average=False) / rec_x.size(0)

            loss = beta * kld_loss + rec_loss
            model.zero_grad()
            loss.backward()
            trainer.step()

            if (b + 1) % 100 == 0:
                print('[ %2d / %2d ] [%5d] kld_loss: %.4f rec_loss: %4f ' % (
                    e + 1, n_epochs, b + 1, kld_loss.item(), rec_loss.item()))

        # save
        if (e + 1) % 5 == 0:
            torch.save(model.state_dict(), save_dir + 'beta_{}_vae_{}.pth'.format(beta, e + 1))
        # test
        with torch.no_grad():
            # 随机生成
            model.eval()
            z = torch.randn(64, model.z_dim).to(DEVICE)
            test_ims = F.sigmoid(model.decoder(z))
            tv.utils.save_image(test_ims, save_dir + 'im_{}.png'.format(e + 1))


def test(model, batch_size, width, save_dir='./results/'):
    # 对隐层的某一维进行插值， 其他维度不变
    model.eval()
    with torch.no_grad():
        for dim in range(model.z_dim):
            z = torch.randn(model.z_dim).repeat(batch_size, 1)
            z[:, dim] = torch.linspace(-width, width, batch_size)
            test_ims = F.sigmoid(model.decoder(z.to(DEVICE)))
            tv.utils.save_image(test_ims, save_dir + 'im_w{}_d{}.png'.format(width, dim))
