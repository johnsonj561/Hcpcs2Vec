import time
from gensim.models.callbacks import CallbackAny2Vec


class GensimEpochCallback(CallbackAny2Vec):
    def __init__(self, loss_out, time_out):
        self.t0 = time.time()
        self.loss = []
        self.times = []
        self.loss_out = loss_out
        self.time_out = time_out
        self.epoch = 0
        self.prev_loss = -1

    def on_epoch_end(self, model):
        delta = time.time() - self.t0
        self.t0 = time.time()
        self.times.append(str(delta))
        cum_loss = model.get_latest_training_loss()
        loss = cum_loss if self.prev_loss < 0 else cum_loss - self.prev_loss
        self.prev_loss = cum_loss
        self.loss.append(str(loss))
        self.epoch += 1
        print(f"Epoch: {self.epoch} \t loss: {loss} \t time: {delta}")

    def on_train_end(self, model):
        with open(self.loss_out, "w") as fout:
            fout.write(",".join(self.loss))
        with open(self.time_out, "w") as fout:
            fout.write(",".join(self.times))
