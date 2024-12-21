import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PhysicsInformedNN:
    def __init__(self, t0, u0, layers, dt, lb, ub):
        self.lb = lb
        self.ub = ub

        self.t0 = t0  # 時間データ
        self.u0 = u0  # 速度データ

        self.layers = layers
        self.dt = dt

        # ニューラルネットワークの初期化
        self.weights, self.biases = self.initialize_NN(layers)
        # オプティマイザ
        self.optimizer = tf.keras.optimizers.Nadam()

        # モデル出力と損失
        self.u_pred = self.net_u(self.t0)
        self.loss = self.get_loss(self.t0, self.u0)

        # 学習履歴
        self.loss_history = []

    def initialize_NN(self, layers):
        """ Xavier初期化で重みとバイアスを生成 """
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = tf.Variable(tf.random.normal([layers[l], layers[l + 1]],
                                             stddev=np.sqrt(2 / (layers[l] + layers[l + 1]))),
                            dtype=tf.float32)
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, X, weights, biases):
        """ ニューラルネットワークの順伝播 """
        H = X
        for l in range(len(weights) - 1):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, t):
        """ 速度 (ux, uy, uz) の予測 """
        return self.neural_net(t, self.weights, self.biases)

    def get_loss(self, t, u):
        """ 損失関数の計算 """
        u_pred = self.net_u(t)
        loss = tf.reduce_mean(tf.square(u - u_pred))
        return loss

    def train(self, nIter):
        """ トレーニングループ """
        for it in range(nIter):
            with tf.GradientTape() as tape:
                loss_value = self.get_loss(self.t0, self.u0)
            gradients = tape.gradient(loss_value, self.weights + self.biases)
            self.optimizer.apply_gradients(zip(gradients, self.weights + self.biases))
            self.loss_history.append(loss_value.numpy())

            if it % 50 == 0:
                print(f"Iteration: {it}, Loss: {loss_value.numpy():.3e}")

    def predict(self, t_star):
        """ 速度の予測 """
        return self.net_u(t_star).numpy()

def load_data(file_path, sheet_name):
    """ Excelからデータを読み込む """
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=2)
    t = df.iloc[:, 0:1].values.astype(np.float32)  # 時間データ (A列)
    u = df.iloc[:, 1:4].values.astype(np.float32)  # 速度データ (B列:ux, C列:uy, D列:uz)
    return t, u

def plot_4_window_results(model, t, u_true, u_pred):
    """ 4窓表示で損失関数と観測データ vs PINNs予測結果を比較するグラフ """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 左上: 損失関数 vs エポック数
    axs[0, 0].plot(range(len(model.loss_history)), model.loss_history, 'r-', label="Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_title("Loss vs Epoch")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # 右上: ux の比較
    axs[0, 1].plot(t, u_true[:, 0], 'b-', label="Observed ux")
    axs[0, 1].plot(t, u_pred[:, 0], 'r--', label="Predicted ux")
    axs[0, 1].set_title("Comparison of ux: Observed vs Predicted")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # 左下: uy の比較
    axs[1, 0].plot(t, u_true[:, 1], 'b-', label="Observed uy")
    axs[1, 0].plot(t, u_pred[:, 1], 'r--', label="Predicted uy")
    axs[1, 0].set_title("Comparison of uy: Observed vs Predicted")
    axs[1, 0].legend()
    axs[1, 0].grid()

    # 右下: uz の比較
    axs[1, 1].plot(t, u_true[:, 2], 'b-', label="Observed uz")
    axs[1, 1].plot(t, u_pred[:, 2], 'r--', label="Predicted uz")
    axs[1, 1].set_title("Comparison of uz: Observed vs Predicted")
    axs[1, 1].legend()
    axs[1, 1].grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "C:/labo/PINNs/A_ball.xlsx"
    t0, u0 = load_data(file_path, sheet_name="1")

    layers = [1, 100, 100, 100, 3]  # 入力1 -> 出力3 (ux, uy, uz)
    dt = 0.01
    lb = t0.min()
    ub = t0.max()

    print("Training on Sheet 1...")
    model = PhysicsInformedNN(t0, u0, layers, dt, lb, ub)
    model.train(nIter=500)

    u_pred = model.predict(t0)

    plot_4_window_results(model, t0, u0, u_pred)
