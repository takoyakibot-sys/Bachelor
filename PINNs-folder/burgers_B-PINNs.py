import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow 2.xでEager Executionを無効化し、1.xモードに切り替え
tf.compat.v1.disable_eager_execution()
tfd = tfp.distributions  # TensorFlow Probability の呼び出し

class BayesianPhysicsInformedNN:
    def __init__(self, x, y, layers):
        """ 初期化: 入力データ、ネットワーク構造、オプティマイザを設定 """
        self.x = x
        self.y = y
        self.layers = layers

        # プレースホルダーの設定
        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.phi_true_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        # ベイズニューラルネットワークの構築
        self.weights, self.biases = self.initialize_NN(layers)
        self.phi_pred, self.u_pred, self.v_pred = self.neural_net(self.x_tf, self.y_tf)

        # 損失関数とベイズ推論: 正規分布に基づく誤差モデル
        self.loss = tf.reduce_mean(tf.square(self.phi_true_tf - self.phi_pred))
        self.neg_log_likelihood = -tf.reduce_sum(tfd.Normal(loc=self.phi_pred, scale=1.0).log_prob(self.phi_true_tf))

        # オプティマイザの設定 (Adam)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = self.optimizer.minimize(self.neg_log_likelihood)

        # TensorFlowのセッション
        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        """ ネットワークの重みとバイアスの初期化 """
        weights = []
        biases = []
        for l in range(len(layers) - 1):
            W = tf.Variable(tf.random.truncated_normal([layers[l], layers[l + 1]], stddev=0.1), dtype=tf.float32)
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, x, y):
        """ ニューラルネットワークの構築 """
        num_layers = len(self.weights)
        H = tf.concat([x, y], axis=1)  # 入力層: x, y
        for l in range(num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        output = tf.add(tf.matmul(H, W), b)
        phi = output[:, 0:1]  # ポテンシャル
        u = output[:, 1:2]    # 速度u
        v = output[:, 2:3]    # 速度v
        return phi, u, v

    def train(self, x_train, y_train, phi_true, epochs):
        """ 学習ループ """
        for epoch in range(epochs):
            tf_dict = {self.x_tf: x_train, self.y_tf: y_train, self.phi_true_tf: phi_true}
            self.sess.run(self.train_op, tf_dict)
            if epoch % 100 == 0:
                neg_log_likelihood_value = self.sess.run(self.neg_log_likelihood, tf_dict)
                print(f"Epoch: {epoch}, Negative Log-Likelihood: {neg_log_likelihood_value:.3e}")

    def predict(self, x_star, y_star):
        """ 予測: ポテンシャル、速度u, v を計算 """
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}
        phi_star = self.sess.run(self.phi_pred, tf_dict)
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        return phi_star, u_star, v_star

# データの読み込み
data = pd.read_csv("C:\\labo\\GPT\\flow_data.csv")
x = data['x'].values[:, None]
y = data['y'].values[:, None]
phi_true = np.sin(np.pi * x) * np.sin(np.pi * y)  # 初期条件: 真のポテンシャル (テスト用)

# モデルの設定と学習
layers = [2, 50, 50, 50, 3]  # 入力: (x, y), 出力: (φ, u, v)
b_pinns = BayesianPhysicsInformedNN(x, y, layers)
b_pinns.train(x, y, phi_true, epochs=1000)

# 予測と保存
phi_pred, u_pred, v_pred = b_pinns.predict(x, y)
va = np.sqrt(u_pred**2 + v_pred**2)  # 予測合成速度

# 観測データの定義 (初期条件に合わせる: 真の速度成分と合成速度)
u_true = np.zeros_like(x)  # uの観測値はゼロで初期化
v_true = np.zeros_like(y)  # vの観測値はゼロで初期化
va_true = np.sqrt(u_true**2 + v_true**2)

# 観測データと学習データを比較する4窓グラフ
fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=100)
plt.rcParams["font.size"] = 15

# x vs u
axes[0, 0].scatter(x, u_pred, color='blue', label='Predicted u', s=10)
axes[0, 0].scatter(x, u_true, color='red', label='Observed u', s=10)
axes[0, 0].set_title('x vs u')
axes[0, 0].legend()

# x vs v
axes[0, 1].scatter(x, v_pred, color='blue', label='Predicted v', s=10)
axes[0, 1].scatter(x, v_true, color='red', label='Observed v', s=10)
axes[0, 1].set_title('x vs v')
axes[0, 1].legend()

# x vs phi
axes[1, 0].scatter(x, phi_pred, color='blue', label='Predicted φ', s=10)
axes[1, 0].scatter(x, phi_true, color='red', label='Observed φ', s=10)
axes[1, 0].set_title('x vs φ')
axes[1, 0].legend()

# x vs velocity magnitude
axes[1, 1].scatter(x, va, color='blue', label='Predicted |v|', s=10)
axes[1, 1].scatter(x, va_true, color='red', label='Observed |v|', s=10)
axes[1, 1].set_title('x vs Velocity Magnitude')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
