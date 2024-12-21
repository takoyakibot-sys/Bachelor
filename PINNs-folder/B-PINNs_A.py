# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:53:14 2024

@author: kotaro
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow 2.xでEager Executionを無効化し、1.xモードに切り替え
tf.compat.v1.disable_eager_execution()
tfd = tfp.distributions  # TensorFlow Probability の呼び出し

class BayesianPhysicsInformedNN:
    # クラスの初期化
    def __init__(self, t0, u0, layers, dt, lb, ub):
        self.lb = lb
        self.ub = ub
        
        # 初期データ
        self.t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
        self.u0 = tf.convert_to_tensor(u0, dtype=tf.float32)
        
        self.layers = layers
        self.dt = dt
        
        # ニューラルネットワークの初期化
        self.weights, self.biases = self.initialize_NN(layers)
        # オプティマイザ (NAdam)
        self.optimizer = tf.keras.optimizers.Nadam()
        
        # 損失関数: ベイズ的負の対数尤度
        self.loss = self.get_loss(self.t0, self.u0)
        
        # 最適化操作 (train_op)
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
        
        # セッションの初期化
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # 学習履歴
        self.loss_history = []

    def initialize_NN(self, layers):
        """ Xavier初期化で重みとバイアスを生成 """
        weights = []
        biases = []
        for l in range(len(layers)-1):
            W = tf.Variable(tf.random.normal([layers[l], layers[l+1]], stddev=np.sqrt(2 / (layers[l] + layers[l+1]))), dtype=tf.float32)
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, X, weights, biases):
        """ ニューラルネットワークの順伝播 """
        H = X
        for l in range(len(weights)-1):
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
        """ 損失関数: ベイズ的負の対数尤度 """
        u_pred = self.net_u(t)
        likelihood = tfd.Normal(loc=u_pred, scale=0.1)
        neg_log_likelihood = -tf.reduce_mean(likelihood.log_prob(u))
        return neg_log_likelihood

    def train(self, nIter):
        """ トレーニングループ """
        for it in range(nIter):
            _, loss_value = self.sess.run([self.train_op, self.loss])
            self.loss_history.append(loss_value)
            
            if it % 50 == 0:
                print(f"Iteration: {it}, Negative Log-Likelihood: {loss_value:.3e}")

    def predict(self, t_star):
        """ 速度の予測 """
        t_star = tf.convert_to_tensor(t_star, dtype=tf.float32)
        u_star = self.sess.run(self.net_u(t_star))
        return u_star

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
    axs[0, 0].set_ylabel("Negative Log-Likelihood")
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
    # シート「1」のデータを読み込む
    file_path = "C:/labo/PINNs/A_ball.xlsx"
    t0, u0 = load_data(file_path, sheet_name="1")
    
    # モデルの設定
    layers = [1, 100, 100, 100, 3]  # 入力1 -> 出力3 (ux, uy, uz)
    dt = 0.01  # 時間ステップ
    lb = t0.min()   # 下限
    ub = t0.max()   # 上限

    # B-PINNsの学習
    print("Training on Sheet 1 with B-PINNs...")
    model = BayesianPhysicsInformedNN(t0, u0, layers, dt, lb, ub)
    model.train(nIter=500)
    
    # 予測結果
    u_pred = model.predict(t0)
    
    # 4窓表示でグラフを表示
    plot_4_window_results(model, t0, u0, u_pred)
