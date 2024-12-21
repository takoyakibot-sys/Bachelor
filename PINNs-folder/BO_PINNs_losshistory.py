# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:44:39 2024

@author: kotaro
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Integer

class PhysicsInformedNN(tf.keras.Model):
    def __init__(self, layers):
        super(PhysicsInformedNN, self).__init__()
        self.layers_list = []
        for i in range(len(layers) - 1):
            self.layers_list.append(
                tf.keras.layers.Dense(
                    layers[i + 1],
                    activation='tanh' if i < len(layers) - 2 else None,
                    kernel_initializer='glorot_normal'
                )
            )

    def call(self, t):
        """順伝播"""
        x = t
        for layer in self.layers_list:
            x = layer(x)
        return x

    def compute_loss(self, t, u):
        """損失関数の計算"""
        u_pred = self(t)
        data_loss = tf.reduce_mean(tf.square(u - u_pred))
        return data_loss

    def compute_uncertainty(self, t):
        """不確実性の評価"""
        u_pred_samples = [self(t) for _ in range(10)]  # 予測値を複数回計算
        u_pred_stack = tf.stack(u_pred_samples)  # サンプルをスタック
        uncertainty = tf.math.reduce_std(u_pred_stack, axis=0)  # 分散を計算
        return tf.reduce_mean(uncertainty)  # 平均不確実性を返す

    def train_step(self, t, u, optimizer):
        """1ステップのトレーニング"""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(t, u)
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

def load_data(file_path, sheet_name):
    """ Excelからデータを読み込む """
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=2)
    t = df.iloc[:, 0:1].values.astype(np.float32)  # 時間データ (A列)
    u = df.iloc[:, 1:4].values.astype(np.float32)  # 速度データ (B列:ux, C列:uy, D列:uz)
    return tf.convert_to_tensor(t), tf.convert_to_tensor(u)

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

# ベイズ最適化用の目的関数
def objective(params):
    """目的関数: 学習率、エポック数、不確実性を最適化"""
    learning_rate, nIter, lambda_u = params

    # モデルの再構築
    model = PhysicsInformedNN(layers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.loss_history = []

    # トレーニングを実行
    for epoch in range(int(nIter)):
        loss = model.train_step(t0, u0, optimizer)
        model.loss_history.append(float(loss))

    # 最終的な不確実性を計算
    uncertainty = model.compute_uncertainty(t0)

    # 損失と不確実性の加重和を最適化
    total_metric = model.loss_history[-1] + lambda_u * float(uncertainty)
    print(f"Learning rate: {learning_rate}, Epochs: {nIter}, Loss: {model.loss_history[-1]}, Uncertainty: {float(uncertainty)}, Metric: {total_metric}")
    return total_metric

if __name__ == "__main__":
    # データの読み込み
    file_path = "C:/labo/PINNs/A_ball.xlsx"
    t0, u0 = load_data(file_path, sheet_name="1")
    
    # モデルの設定
    layers = [1, 100, 100, 100, 3]

    # ベイズ最適化の設定
    space = [
        Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),  # 学習率の範囲
        Integer(100, 500, name='nIter'),  # エポック数の範囲
        Real(0.1, 10.0, name='lambda_u')  # 不確実性の重み
    ]

    print("Running Bayesian Optimization...")
    result = gp_minimize(objective, space, n_calls=20, random_state=42)

    # 最適なハイパーパラメータを出力
    optimal_learning_rate, optimal_nIter, optimal_lambda_u = result.x
    print(f"Optimal learning rate: {optimal_learning_rate}")
    print(f"Optimal nIter: {optimal_nIter}")
    print(f"Optimal lambda_u: {optimal_lambda_u}")
    print(f"Minimum metric: {result.fun}")

    # 最適なハイパーパラメータで最終学習
    print("Training final model with optimal parameters...")
    model = PhysicsInformedNN(layers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=optimal_learning_rate)
    model.loss_history = []

    for epoch in range(int(optimal_nIter)):
        loss = model.train_step(t0, u0, optimizer)
        model.loss_history.append(float(loss))

    # 予測と可視化
    u_pred = model(t0).numpy()
    plot_4_window_results(model, t0.numpy(), u0.numpy(), u_pred)
