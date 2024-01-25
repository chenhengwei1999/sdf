import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, latent_size * 2)  # mean and log-variance

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, latent_size)
        self.decoder = Decoder(latent_size, input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decoder(z)

        return recon_x, mu, logvar

# 示例用法
input_size = 100  # 替换成实际输入数据的维度
latent_size = 10  # 设置潜在空间的维度
vae = VAE(input_size, latent_size)

# 定义损失函数
def loss_function(recon_x, x, mu, logvar):
    # 重构损失
    reconstruction_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL 散度项
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + kl_divergence

# 定义优化器
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
# for epoch in range(num_epochs):
#     total_loss = 0.0
#     for data in dataloader:
#         inputs = data  # 根据实际情况获取输入数据
#         optimizer.zero_grad()
#         recon_x, mu, logvar = vae(inputs)
#         loss = loss_function(recon_x, inputs, mu, logvar)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     average_loss = total_loss / len(dataloader.dataset)
#     print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss}')

# # 保存训练好的模型
# torch.save(vae.state_dict(), 'vae_model.pth')
