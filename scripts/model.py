import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=6, num_layers=1):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        mu = self.fc_mu(out)
        sigma = torch.exp(self.fc_sigma(out))
        return mu, sigma

class ParallelLSTMs(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(ParallelLSTMs, self).__init__()
        self.lstm_mainbus = LSTMNet(input_dim, hidden_dim, output_dim)
        self.lstm_schlinger = LSTMNet(input_dim, hidden_dim, output_dim)
        self.lstm_resnick = LSTMNet(input_dim, hidden_dim, output_dim)
        self.lstm_beckman = LSTMNet(input_dim, hidden_dim, output_dim)
        self.lstm_braun = LSTMNet(input_dim, hidden_dim, output_dim)
    
    def forward(self, x):
        mainbus_mu, mainbus_sigma = self.lstm_mainbus(x)
        schlinger_mu, schlinger_sigma = self.lstm_schlinger(x)
        resnick_mu, resnick_sigma = self.lstm_resnick(x)
        beckman_mu, beckman_sigma = self.lstm_beckman(x)
        braun_mu, braun_sigma = self.lstm_braun(x)

        mu = torch.cat([mainbus_mu, schlinger_mu, resnick_mu, beckman_mu, braun_mu], dim=1)
        sigma = torch.cat([mainbus_sigma, schlinger_sigma, resnick_sigma, beckman_sigma, braun_sigma], dim=1)

        return mu, sigma