import torch
import torch.optim as optim
from model import ParallelLSTMs
from data_loader import load_data
from cvxpy_protocol import opf_problem_optnn
import numpy as np
from tqdm import tqdm
from utils import str_to_matrices, str_to_vector, load_config, split_real_imag
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set negative log likelihood (NLL) loss function
def nll_loss(mu, sigma, targets):
    return torch.mean(0.5 * torch.log(sigma**2) + 0.5 * (targets - mu)**2 / sigma**2)

# set hyperparameters
data_path = r'C:\Users\11389\OneDrive\桌面\学习资料\科研\python\SURF-main\data\dataset\caltech_data.csv'
n_train_hours = int(0.95 * 113387)

# load data {demand_mainbus,demand_schlinger,demand_resnick,demand_beckman,demand_braun}
train_loader, test_loader, scaler, train_X, train_y, test_X, test_y, n_in, n_out = load_data(data_path, n_train_hours)

# initialize the model, optimizer and loss function
input_dim = 34*n_in
output_dim = 6*n_out
model = ParallelLSTMs(input_dim=input_dim, hidden_dim=256, output_dim=output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nll_loss

train the model
for epoch in range(256):
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for inputs, targets in tepoch:
            # neural network module
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            mu, sigma = model(inputs)
            # print(mainbus_mu.shape, mainbus_sigma.shape, targets.shape)
            loss_pred = criterion(mu, sigma, targets)

            # uncertainty calibration
            pred_low = mu - 2 * sigma
            pred_high = mu + 2 * sigma
            scores = torch.maximum(
                torch.max(pred_low - targets, dim=1)[0],
                torch.max(targets - pred_high, dim=1)[0]
            )
            alpha = 0.05  # 0.95-confidence
            n = len(scores)
            j = int(np.ceil((n+1) * (1-alpha)))
            if j > n: j = n
            sorted_inds = torch.argsort(scores)
            q = scores[sorted_inds[j-1]]

            U_low_cali = pred_low - q
            U_high_cali = pred_high + q

            # inverse transform
            inv_U_high_cali = scaler.inverse_transform(
                np.concatenate((U_high_cali.cpu().numpy(), inputs[:, 0, 30:].cpu().numpy()), axis=1)
            )
            inv_U_high_cali = inv_U_high_cali[:, :30]

            # grid data
            config = load_config("config.yaml")
            Y_A = {}
            for key, value in config['Y_A'].items():
                Y_A[key] = str_to_matrices(value)
            RTP = np.array([config['RTP']])
            T = config['T']
            adj_matrix = np.array(config['adj_matrix'])
            N = len(adj_matrix)
            node_class = config['node_class']

            # two-stage RO optimization module
            demands_mainbus = torch.complex(inv_U_high_cali[:, 0:3], U_high_cali[:, 3:6])
            demands_schlinger = torch.complex(inv_U_high_cali[:, 6:9], U_high_cali[:, 9:12])
            demands_resnick = torch.complex(inv_U_high_cali[:, 12:15], U_high_cali[:, 15:18])
            demands_beckman = torch.complex(inv_U_high_cali[:, 18:21], U_high_cali[:, 21:24])
            demands_braun = torch.complex(inv_U_high_cali[:, 24:27], U_high_cali[:, 27:30])

            real_mainbus, imag_mainbus = split_real_imag(demands_mainbus)
            real_schlinger, imag_schlinger = split_real_imag(demands_schlinger)
            real_resnick, imag_resnick = split_real_imag(demands_resnick)
            real_beckman, imag_beckman = split_real_imag(demands_beckman)
            real_braun, imag_braun = split_real_imag(demands_braun)
            opf = opf_problem_optnn(T, N, Y_A, adj_matrix, node_class, RTP, 
                                    real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                                    imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun)
            loss_opf = opf.torch_loss(real_mainbus, real_schlinger, real_resnick, real_beckman, real_braun,
                                        imag_mainbus, imag_schlinger, imag_resnick, imag_beckman, imag_braun)
            print(loss_opf)
            loss_task = torch.mean(loss_opf)

            asdasd
            loss = loss_pred * 0.99 + loss_task * 0.01
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # update progress bar
            tepoch.set_postfix(loss=total_loss / len(tepoch))

    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

# save the trained model
torch.save(model.state_dict(), r"C:\Users\11389\OneDrive\桌面\学习资料\科研\python\SURF-main\ckpt\ckpt.pth")

# test the model
model.load_state_dict(torch.load(r"C:\Users\11389\OneDrive\桌面\学习资料\科研\python\SURF-main\ckpt\ckpt.pth"))
model.eval()
predictions, error_band, actuals = [], [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        mu, sigma = model(inputs)
        predictions.append(mu.cpu())
        actuals.append(targets.cpu())
        error_band.append(sigma.cpu())

predictions = torch.cat(predictions).numpy()
actuals = torch.cat(actuals).numpy()
error_band = torch.cat(error_band).numpy()
pred_low = predictions - 1*error_band
pred_high = predictions + 1*error_band

a = torch.from_numpy(predictions)
min_val = torch.tensor(scaler.min_, dtype=torch.float32, device=a.device)
scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=a.device)
print(scaler.min_, scaler.scale_)
combined_tensor = torch.cat((a, test_X[:, 0, 30:]), dim=1)
inv_a = (combined_tensor - min_val)/scale
inv_a = inv_a[:, :30]

inv_predictions = scaler.inverse_transform(
    np.concatenate((predictions, test_X[:, 0, 30:].cpu().numpy()), axis=1)
)
inv_actuals = scaler.inverse_transform(
    np.concatenate((actuals, test_X[:, 0, 30:].cpu().numpy()), axis=1)
)
inv_pred_low = scaler.inverse_transform(
    np.concatenate((pred_low, test_X[:, 0, 30:].cpu().numpy()), axis=1)
)
inv_pred_high = scaler.inverse_transform(
    np.concatenate((pred_high, test_X[:, 0, 30:].cpu().numpy()), axis=1)
)

inv_predictions = inv_predictions[0:300, :30]
print(inv_a.shape, inv_predictions.shape)
print(inv_a[0:30, 0])
print(inv_predictions[0:30, 0])
inv_actuals = inv_actuals[0:300, :30]
inv_pred_low = inv_pred_low[0:300, :30]
inv_pred_high = inv_pred_high[0:300, :30]


plt.figure(figsize=(10, 6))
plt.plot(inv_actuals[:, 0], label='Ground-truth real part 1', color='#ffbf00')
plt.plot(inv_predictions[:, 0], label='Predicted real part 1', color='#1070bf')
plt.fill_between(range(len(inv_predictions)), inv_pred_low[:, 0], inv_pred_high[:, 0], color='#1070bf', alpha=0.2)
plt.legend()
plt.show()
