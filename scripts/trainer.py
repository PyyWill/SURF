import torch
import torch.optim as optim
from model import ParallelLSTMs
from data_loader import load_data
from cvxpy_protocol import OptimalPowerFlowNeuralNetwork
import numpy as np
from tqdm import tqdm
from utils import str_to_matrices, str_to_vector, load_config, split_real_imag
import matplotlib.pyplot as plt

def torch_inverse_transform(x_scaled, scaler):
    scale = torch.tensor(scaler.scale_, dtype=x_scaled.dtype, device=x_scaled.device)
    min_ = torch.tensor(scaler.min_, dtype=x_scaled.dtype, device=x_scaled.device)
    return x_scaled * scale + min_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set negative log likelihood (NLL) loss function
def nll_loss(mu, sigma, targets):
    return torch.mean(0.5 * torch.log(sigma**2) + 0.5 * (targets - mu)**2 / sigma**2)

# set hyperparameters
data_path = r'C:\PythonProject\SURF\data\power_1014-0114_with_weather.csv'
n_train_hours = int(0.9 * 1275)

# load data {demand_mainbus,demand_broad,demand_schlinger,demand_resnick,demand_beckman,demand_braun}
train_loader, test_loader, scaler, train_X, train_y, test_X, test_y, n_in, n_out = load_data(data_path, n_train_hours)

# initialize the model, optimizer and loss function
input_dim = 41*n_in  # 6 variables * 2 parts * 3 phases + 5 weather = 36 + 5 = 41 features
output_dim = 6*n_out  # 6 variables (mainbus, broad, schlinger, resnick, beckman, braun)
model = ParallelLSTMs(input_dim=input_dim, hidden_dim=256, output_dim=output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nll_loss

# train the model
for epoch in range(2):
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

            # # uncertainty calibration
            # pred_low = mu - 2 * sigma
            # pred_high = mu + 2 * sigma
            # scores = torch.maximum(
            #     torch.max(pred_low - targets, dim=1)[0],
            #     torch.max(targets - pred_high, dim=1)[0]
            # )
            # alpha = 0.05  # 0.95-confidence
            # n = len(scores)
            # j = int(np.ceil((n+1) * (1-alpha)))
            # if j > n: j = n
            # sorted_inds = torch.argsort(scores)
            # q = scores[sorted_inds[j-1]]

            # U_low_cali = pred_low - q
            # U_high_cali = pred_high + q

            # # inverse transform
            # U_high_cali_cat = torch.cat((U_high_cali, inputs[:, 0, 36:]), dim=1)  # 拼接保持在 torch 内部
            # inv_U_high_cali = torch_inverse_transform(U_high_cali_cat, scaler)
            # inv_U_high_cali = inv_U_high_cali[:, :36]

            # # grid data
            # config = load_config("config.yaml")
            # Y_A = {}
            # for key, value in config['Y_A'].items():
            #     Y_A[key] = str_to_matrices(value)
            # RTP = float(3.2939005e-05)
            # T = config['T']
            # adj_matrix = np.array(config['adj_matrix'])
            # N = len(adj_matrix)
            # node_class = config['node_class']

            # # two-stage RO optimization module
            # slices = {
            #     "real_mainbus":   (0, 3),
            #     "real_broad":     (6, 9),
            #     "real_schlinger": (12, 15),
            #     "real_resnick":   (18, 21),
            #     "real_beckman":   (24, 27),
            #     "real_braun":     (30, 33),
            # }

            # reals = {}
            # valid = True
            # for name, (start, end) in slices.items():
            #     tensor = inv_U_high_cali[:, start:end]
            #     if tensor.shape != (2, 3):
            #         print(f"Skip batch: {name} shape {tensor.shape}, expected (2,3)")
            #         valid = False
            #         break
            #     reals[name] = tensor

            # if not valid:
            #     continue 

            # # Create OPF problem with only real parameters
            # opf = OptimalPowerFlowNeuralNetwork(
            #     T, N, Y_A, adj_matrix, node_class, RTP,
            #     reals["real_mainbus"], reals["real_broad"], reals["real_schlinger"],
            #     reals["real_resnick"], reals["real_beckman"], reals["real_braun"]
            # )

            # loss_opf = opf.torch_loss(reals["real_mainbus"], reals["real_broad"], reals["real_schlinger"], 
            #                          reals["real_resnick"], reals["real_beckman"], reals["real_braun"])

            loss = loss_pred
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # update progress bar
            tepoch.set_postfix(loss=total_loss / len(tepoch))

# save the trained model
torch.save(model.state_dict(), "ckpt.pth")

# test the model (use the already trained model)
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
combined_tensor = torch.cat((a, test_X[:, 0, 36:]), dim=1)
inv_a = (combined_tensor - min_val)/scale
inv_a = inv_a[:, :36]

inv_predictions = scaler.inverse_transform(
    np.concatenate((predictions, test_X[:, 0, 36:].cpu().numpy()), axis=1)
)
inv_actuals = scaler.inverse_transform(
    np.concatenate((actuals, test_X[:, 0, 36:].cpu().numpy()), axis=1)
)
inv_pred_low = scaler.inverse_transform(
    np.concatenate((pred_low, test_X[:, 0, 36:].cpu().numpy()), axis=1)
)
inv_pred_high = scaler.inverse_transform(
    np.concatenate((pred_high, test_X[:, 0, 36:].cpu().numpy()), axis=1)
)

inv_predictions = inv_predictions[0:300, :18]  # 6 variables * 3 phases = 18
inv_actuals = inv_actuals[0:300, :18]
inv_pred_low = inv_pred_low[0:300, :18]
inv_pred_high = inv_pred_high[0:300, :18]


plt.figure(figsize=(10, 6))
plt.plot(inv_actuals[:, 0], label='Ground-truth real part 1', color='#ffbf00')
plt.plot(inv_predictions[:, 0], label='Predicted real part 1', color='#1070bf')
plt.fill_between(range(len(inv_predictions)), inv_pred_low[:, 0], inv_pred_high[:, 0], color='#1070bf', alpha=0.2)
plt.legend()
plt.show()
