import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from PPdata import training_set, test_set
import numpy as np
from torch.utils.data import DataLoader
import ranger

args_dict = {
    "input_size": 1,
    "hidden_layer_size": 32,
    "num_layers": 2,
    "output_size": 1,
    "dropout": 0.7,
    "batch_size": 64,
    "step_size": 40,
    "lr": 0.1,
    "epoch": 100
}

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        batchsize = x.shape[0]

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:,-1]

#  define the LSTM model
model = LSTMModel(input_size=args_dict["input_size"],
                  hidden_layer_size=args_dict["hidden_layer_size"],
                  num_layers=args_dict["num_layers"],
                  output_size=args_dict["output_size"],
                  dropout=args_dict["dropout"])


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
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
optimizer = ranger.Ranger(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

#training
for epoch in range(args_dict["epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch+1, args_dict["epoch"], loss_train, loss_val, lr_train))


train_dataloader = DataLoader(training_set, batch_size=args_dict["batch_size"], shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=args_dict["batch_size"], shuffle=False)

model.eval()


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

plt.figure()
plt.plot(test_set.y, label="actual")
plt.plot(predicted_test, label="predicted")
plt.title("test set")
plt.legend()

plt.figure()
plt.plot(training_set.y, label="actual")
plt.plot(predicted_train, label="predicted")
plt.title("training set")
plt.legend()
