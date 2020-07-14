import os

import torch

from beta_vae_h import test, BetaVAE_H, DEVICE

z_dim = 16
beta = 4
save_dir = './b4z16/'
os.makedirs(save_dir, exist_ok=True)

model = BetaVAE_H(z_dim, 3).to(DEVICE)
# train(model,
#       data_iter=chairs_3d_iter,
#       lr=1e-4,
#       n_epochs=1000,
#       beta=beta,
#       save_dir=save_dir)

model.load_state_dict(torch.load('beta_4_vae_1000.pth'))

for w in [3, 1.5]:
    test(model,
         batch_size=8,
         width=w,
         save_dir=save_dir)
