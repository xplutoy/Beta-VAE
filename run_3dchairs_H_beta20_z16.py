import os

from beta_vae_h import train, test, BetaVAE_H, DEVICE
from data import chairs_3d_iter

z_dim = 16
beta = 20
save_dir = './b20z16/'
os.makedirs(save_dir, exist_ok=True)

model = BetaVAE_H(z_dim, 3).to(DEVICE)
train(model,
      data_iter=chairs_3d_iter,
      lr=1e-3,
      n_epochs=50,
      beta=beta,
      save_dir=save_dir)

# model.load_state_dict(torch.load('beta_4_vae_1000.pth'))

for w in [3, 1.5]:
    test(model,
         batch_size=8,
         width=w,
         save_dir=save_dir)
