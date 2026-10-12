from pathlib import Path

from scipy.io import loadmat

data_path = Path(__file__).resolve().parent.parent / "data" / "SNLS80mV.mat"
data = loadmat(data_path)

u_all = data["V1"].flatten()
y_all = data["V2"].flatten()

start = 42650
stop = 46746
u = u_all[start:stop]
y = y_all[start:stop]

train_size = 3072
u_train = u[:train_size]
u_test = u[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

u_mean = u.mean()
y_mean = y.mean()
u_train_centered = u_train - u_mean
u_test_centered = u_test - u_mean
y_train_centered = y_train - y_mean
y_test_centered = y_test - y_mean

print("Selected shapes:", u.shape, y.shape)
print("Training shapes:", u_train.shape, y_train.shape)
print("Test shapes:", u_test.shape, y_test.shape)
print("Centered training means:", u_train_centered.mean(), y_train_centered.mean())
