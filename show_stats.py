import os

import matplotlib.pyplot as plt
import pandas
from matplotlib import style

from config import BASE_DIR

style.use("ggplot")

def create_acc_loss_graph(log_name):
  contents = pandas.read_csv(os.path.join(BASE_DIR, log_name), names=["epoch", "it", "loss", "test_loss", "lr"])
  contents = contents[contents["loss"].between(0,250)]
  contents = contents[contents["test_loss"].between(0,250)]
  contents["loss"] = contents.rolling(window=50)["loss"].mean()
  contents["loss_avg"] = contents.rolling(window=500)["loss"].mean()
  contents["test_loss"] = contents.rolling(window=50)["test_loss"].mean()
  contents["test_loss_avg"] = contents.rolling(window=500)["test_loss"].mean()
  contents["lr"] = contents.rolling(window=50)["lr"].mean()
  contents.plot(y=["loss", "test_loss", "loss_avg", "test_loss_avg"])
  plt.show()

  # epochs = []
  # its = []
  # losses = []
  # test_losses = []
  #
  # for c in contents:
  #   if c:
  #     epoch, it, loss, test_loss = c.split(",")
  #
  #     epochs.append(int(epoch))
  #     its.append(int(it))
  #     losses.append(float(loss))
  #     test_losses.append(float(test_loss))
  #
  # fig = plt.figure()
  # ax1 = plt.subplot2grid((1,1), (0,0))
  # # ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
  #
  # ax1.plot(losses, label="losses")
  # ax1.plot(test_losses, label="test_losses")
  # ax1.legend(loc=2)
  #
  # # ax2.plot(times, losses, label="loss")
  # # ax2.plot(times, test_losses, label="test_loss")
  # # ax2.legend(loc=2)
  #
  # fig.show()

# batch size
create_acc_loss_graph("log-model-b8-0.001-1623160593.txt")  # bad results, probably too high lr for this bs
create_acc_loss_graph("log-model-b16-0.001-1623147822.txt")  # bad results, probably too high lr for this bs
# create_acc_loss_graph("log-model-b32-0.001-1623101957.txt")
create_acc_loss_graph("log-model-b32-0.001-1623102815.txt")
create_acc_loss_graph("log-model-b64-0.001-1623147811.txt")
create_acc_loss_graph("log-model-b128-0.001-1623103067.txt")
create_acc_loss_graph("log-model-b384-0.001-1623116338.txt")
create_acc_loss_graph("log-model-b384-0.001-1623119461.txt")


# lr
# # create_acc_loss_graph("log-model-b32-0.1-1623160431.txt") # too high
# # create_acc_loss_graph("log-model-b32-0.01-1623157179.txt") # too high
# # create_acc_loss_graph("log-model-b32-0.005-1623162572.txt") # too high
# create_acc_loss_graph("log-model-b32-0.001-1623102815.txt")
# create_acc_loss_graph("log-model-b32-0.0001-1623147850.txt")
# create_acc_loss_graph("log-model-b32-5e-05-1623160526.txt")
# create_acc_loss_graph("log-model-b32-1e-05-1623147883.txt")
# create_acc_loss_graph("log-model-b32-1e-05-1623188043.txt") # 100 epochs
# lr 0.0001 seems best


# lr bs test. reference: bs 32 lr 0.001
create_acc_loss_graph("log-model-b4-0.000125-1623188634.txt")
create_acc_loss_graph("log-model-b8-0.00025-1623188601.txt")
create_acc_loss_graph("log-model-b16-0.0005-1623188587.txt")
create_acc_loss_graph("log-model-b32-0.001-1623102815.txt")
create_acc_loss_graph("log-model-b64-0.002-1623188567.txt")