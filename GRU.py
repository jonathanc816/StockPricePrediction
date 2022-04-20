import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from PPdata import training_set, test_set
import numpy as np
from torch.utils.data import DataLoader
import ranger
from PPdata import scaler, aapl_time, appl_price
import matplotlib.dates as md

args_dict = {
    "input_size": 1,
    "hidden_layer_size": 20,
    "num_layers": 2,
    "output_size": 1,
    "dropout": 0.8,
    "batch_size": 128,
    "step_size": 40,
    "lr": 0.1,
    "epoch": 100
}

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batchsize = x.shape[0]

        # GRU layer
        lstm_out, h_n = self.gru(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.dropout(x)
        predictions = self.linear(x)
        return predictions[:,-1]

#  define the LSTM model
model = GRUModel(input_dim=args_dict["input_size"],
                  hidden_dim=args_dict["hidden_layer_size"],
                  output_dim=args_dict["output_size"],
                  dropout=args_dict["dropout"])


def run(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for index, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr

# create DataLoaders
train_dataloader = DataLoader(training_set, batch_size=args_dict["batch_size"], shuffle=True)
val_dataloader = DataLoader(test_set, batch_size=args_dict["batch_size"], shuffle=True)

# define optimizer (either Ranger or Adam), scheduler and MSE loss function
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9)
optimizer = ranger.Ranger(model.parameters(), lr=args_dict["lr"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args_dict["step_size"], gamma=0.1)

#training
loss_train_lst = []
loss_val_lst = []
for epoch in range(args_dict["epoch"]):
    loss_train, lr_train = run(train_dataloader, is_training=True)
    loss_val, lr_val = run(val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch+1, args_dict["epoch"], loss_train, loss_val, lr_train))
    loss_train_lst.append(loss_train)
    loss_val_lst.append(loss_val)


train_dataloader = DataLoader(training_set, batch_size=args_dict["batch_size"], shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=args_dict["batch_size"], shuffle=False)

model.eval()


plt.figure()
plt.title("Training vs. Validation Losses")
plt.plot(loss_train_lst, label="Train_loss")
plt.plot(loss_val_lst, label="Val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()


#Training set
predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    out = model(x)
    out = out.detach().numpy()
    predicted_train = np.concatenate((predicted_train, out))

#Test set

predicted_test = np.array([])

for idx, (x, y) in enumerate(test_dataloader):
    out = model(x)
    out = out.detach().numpy()
    predicted_test = np.concatenate((predicted_test, out))



inversed_test = scaler.inverse_transform(test_set.y.reshape(test_set.y.shape[0], 1))
inversed_predicted_test = scaler.inverse_transform(predicted_test.reshape(-1,1))

inversed_train = scaler.inverse_transform(training_set.y.reshape(training_set.y.shape[0], 1))
inversed_predicted_train = scaler.inverse_transform(predicted_train.reshape(-1,1))


plt.figure()
formatter = md.DateFormatter("%Y")
locator = md.YearLocator()
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)
plt.plot(aapl_time[:len(inversed_train)], inversed_train, label="actual")
plt.plot(aapl_time[:len(inversed_predicted_train)], inversed_predicted_train, label="predicted")
plt.title("Training set prediction")
plt.xlabel('year')
plt.ylabel('closing price')
plt.legend()

plt.figure()
formatter = md.DateFormatter("%Y")
locator = md.YearLocator()
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)
plt.plot(aapl_time[-len(inversed_test):], inversed_test, label="actual")
plt.plot(aapl_time[-len(inversed_predicted_test):], inversed_predicted_test, label="predicted")
plt.title("Test set prediction\nLast 500 days")
plt.xlabel('year')
plt.ylabel('closing price')
plt.legend()


plt.figure()
formatter = md.DateFormatter("%Y")
locator = md.YearLocator()
ax = plt.gca()
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(locator)
plt.plot(aapl_time[:-len(inversed_test)], appl_price[:-len(inversed_test)], label="Actual data (past)")
plt.plot(aapl_time[-len(inversed_test):], inversed_test, label="Actual price")
plt.plot(aapl_time[-len(inversed_predicted_test):], inversed_predicted_test, label="Predicted price")
plt.title("GRU prediction on AAPL")
plt.xlabel('year')
plt.ylabel('closing price')
plt.legend()
