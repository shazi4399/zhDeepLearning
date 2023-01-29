import torch


#data
import numpy as np
import re
ff = open("housing.data").readlines()
#print(ff)
data = []
for item in ff:
    out = re.sub(r"\s{2,}", " ", item).strip()
    #print(out)
    data.append(out.split(" "))
    #print(data)
data = np.array(data).astype(float)
#print(data.shape)
Y = data[:,-1]
X = data[:,0:-1]

X_train = X[0:496, ...]
Y_train = Y[0:496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]

print(X_train.shape)
print(Y_train.shape)
print(Y_train)

#model(net)
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)

        return out
net = Net(13, 1)

#loss
loss_func = torch.nn.MSELoss()

#optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

#training
for i in range(10000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data)*0.001

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("ite:{}, train_loss:{}".format(i, loss))
    print(pred[0:10])
    print(y_data[0:10])


    #test
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss_test = loss_func(pred, y_data)*0.001
    print("ite:{}, test_loss:{}".format(i, loss_test))

torch.save(net, "mymodel/model.pkl")
#torch.load("")
#torch.save(net.state_dict(), "params.pkl")
#net.load_state_dict("")

