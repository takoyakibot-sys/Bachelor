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
                    activation='relu' if i < len(layers) - 2 else None,
                    kernel_initializer='glorot_uniform'
                )
            )

    def call(self, t):
        """順伝播"""
        x = t
        for layer in self.layers_list:
            x = layer(x)
        return x

    def compute_loss(self, t, u, λ1, λ2):
        """総合損失関数の計算"""
        u_pred = self(t)
        if tf.reduce_any(tf.math.is_nan(u_pred)):
            raise ValueError("Model output (u_pred) contains NaN.")

        # データ損失
        Loss_data = tf.reduce_mean(tf.square(u - u_pred))
        if tf.math.is_nan(Loss_data):
            raise ValueError("Loss_data contains NaN.")

        # 物理法則損失
        with tf.GradientTape() as tape:
            tape.watch(t)
            u_pred = self(t)
            u_t = tape.gradient(u_pred, t)

        if u_t is None or tf.reduce_any(tf.math.is_nan(u_t)):
            raise ValueError("Gradient calculation returned NaN.")

        Loss_physics = tf.reduce_mean(tf.square(u_t - 0.1 * u_pred))
        if tf.math.is_nan(Loss_physics):
            raise ValueError("Loss_physics contains NaN.")

        # 不確実性損失
        u_pred_samples = [self(t) for _ in range(10)]
        u_pred_stack = tf.stack(u_pred_samples)
        Loss_uncertainty = tf.reduce_mean(tf.math.reduce_std(u_pred_stack, axis=0))
        if tf.math.is_nan(Loss_uncertainty):
            raise ValueError("Loss_uncertainty contains NaN.")

        # 総合損失
        total_loss = Loss_data + λ1 * Loss_physics + λ2 * Loss_uncertainty
        if tf.math.is_nan(total_loss):
            raise ValueError("Total loss contains NaN.")

        return total_loss, Loss_data, Loss_physics, Loss_uncertainty

    def train_step(self, t, u, optimizer, λ1, λ2):
        """1ステップのトレーニング"""
        with tf.GradientTape() as tape:
            total_loss, Loss_data, Loss_physics, Loss_uncertainty = self.compute_loss(t, u, λ1, λ2)
        grads = tape.gradient(total_loss, self.trainable_variables)
        grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]  # 勾配のクリッピング
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return total_loss, Loss_data, Loss_physics, Loss_uncertainty

def normalize_data(t, u):
    """データを正規化"""
    t_mean, t_std = np.mean(t), np.std(t)
    u_mean, u_std = np.mean(u, axis=0), np.std(u, axis=0)

    t_normalized = (t - t_mean) / t_std
    u_normalized = (u - u_mean) / u_std

    return t_normalized, u_normalized, t_mean, t_std, u_mean, u_std

def load_data(file_path, sheet_name):
    """ Excelからデータを読み込む """
    df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=2)
    t = df.iloc[:, 0:1].values.astype(np.float32)  # 時間データ (A列)
    u = df.iloc[:, 1:4].values.astype(np.float32)  # 速度データ (B列:ux, C列:uy, D列:uz)

    # NaN チェックと削除
    if np.isnan(t).any() or np.isnan(u).any():
        print("Input data contains NaN values. Cleaning the data...")
        mask = ~np.isnan(t).flatten() & ~np.isnan(u).any(axis=1)
        t = t[mask]
        u = u[mask]

    return t, u

def objective(params, t, u, layers):
    """ベイズ最適化用の目的関数"""
    learning_rate, λ1, λ2 = params

    # モデルの構築
    model = PhysicsInformedNN(layers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # トレーニングループ
    for epoch in range(100):  # エポック数は短縮
        try:
            total_loss, Loss_data, Loss_physics, Loss_uncertainty = model.train_step(t, u, optimizer, λ1, λ2)
        except ValueError as e:
            print(f"Error during training: {e}")
            return 1e6  # エラーが発生した場合、大きな損失を返す

    return float(total_loss)

if __name__ == "__main__":
    # データの読み込み
    file_path = "C:/labo/PINNs/A_ball.xlsx"
    t0, u0 = load_data(file_path, sheet_name="1")
    t0, u0, t_mean, t_std, u_mean, u_std = normalize_data(t0, u0)
    t0 = tf.convert_to_tensor(t0, dtype=tf.float32)
    u0 = tf.convert_to_tensor(u0, dtype=tf.float32)

    # モデルの設定
    layers = [1, 100, 100, 100, 3]

    # ベイズ最適化の設定
    space = [
        Real(1e-5, 1e-3, prior='log-uniform', name='learning_rate'),
        Real(0.1, 10.0, name='λ1'),
        Real(0.1, 10.0, name='λ2')
    ]

    print("[INFO] Running Bayesian Optimization...")
    result = gp_minimize(
        lambda params: objective(params, t0, u0, layers),
        space,
        n_calls=20,
        random_state=42
    )
    print("[INFO] Bayesian Optimization completed.")

    # 最適パラメータ
    optimal_learning_rate, optimal_λ1, optimal_λ2 = result.x
    print(f"Optimal learning rate: {optimal_learning_rate}")
    print(f"Optimal λ1: {optimal_λ1}")
    print(f"Optimal λ2: {optimal_λ2}")
    print(f"Minimum loss: {result.fun}")

    # 最終モデル訓練
    print("[INFO] Training final model with optimal parameters...")
    model = PhysicsInformedNN(layers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=optimal_learning_rate)
    model.loss_history = []

    for epoch in range(500):
        total_loss, Loss_data, Loss_physics, Loss_uncertainty = model.train_step(t0, u0, optimizer, optimal_λ1, optimal_λ2)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss.numpy():.4f}, "
                  f"Data Loss = {Loss_data.numpy():.4f}, "
                  f"Physics Loss = {Loss_physics.numpy():.4f}, "
                  f"Uncertainty Loss = {Loss_uncertainty.numpy():.4f}")

    # 結果表示
    u_pred = model(t0).numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(t0.numpy(), u0.numpy(), label="True")
    plt.plot(t0.numpy(), u_pred, label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.title("PINNs Prediction vs True Data")
    plt.grid()
    plt.show()
