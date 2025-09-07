import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
# from _tkinter import TclError

# CUDA（GPU）が利用可能か確認し、設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # GPUメモリの使用量を動的に設定
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # メモリ増加設定が完了する前に実行するとエラーになる可能性がある
        print(e)
else:
    print("GPUは利用できません。CPUで実行します。")


random.seed(42)

# --- 1. データのロードと前処理 ---

def load_and_preprocess_data(filename="rope_deformation_data.npz", test_data=False):
    """
    Load the collected data and prepare it for model training or evaluation.
    
    Args:
        filename (str): The path to the data file.
        test_data (bool): If True, loads a specific test data set without splitting.
    
    Returns:
        tuple: A tuple containing the prepared data.
    """
    print(f"データのロードを開始します... ({'テストデータ' if test_data else '学習データ'})")
    try:
        loaded_data = np.load(filename, allow_pickle=True)
        pre_state = loaded_data['pre_state']
        post_state = loaded_data['post_state']
        applied_impulses = loaded_data['applied_impulses']
        moved_link_index = loaded_data['moved_link_index']
        
        print("データのロードが完了しました。")
        print(f"力の加える前の状態のデータの形状: {pre_state.shape}")
        print(f"力の加えた後の状態のデータの形状: {post_state.shape}")
        print(f"加えた力積のデータの形状: {applied_impulses.shape}")
        print(f"動かされたリンクのインデックス: {moved_link_index}")

    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。")
        return None, None, None, None

    # 特徴量とラベルを構築
    num_data, num_links, _ = pre_state.shape
    
    node_features = pre_state.reshape(num_data, num_links, 3)
    
    action_features = np.zeros((num_data, num_links, 3))
    for i in range(num_data):
        action_features[i, moved_link_index] = applied_impulses[i]

    X = np.concatenate([node_features, action_features], axis=-1)
    y = post_state.reshape(num_data, num_links, 3)
    
    adj_matrix = np.zeros((num_links, num_links), dtype=np.float32)
    for i in range(num_links - 1):
        adj_matrix[i, i+1] = 1.0
        adj_matrix[i+1, i] = 1.0

    if test_data:
        # テストデータの場合は分割せず、そのまま返す
        adj_matrix_tiled = np.tile(adj_matrix[np.newaxis, :, :], (num_data, 1, 1))
        return (X, y, adj_matrix_tiled), num_links, moved_link_index
    else:
        # 学習データの場合は訓練とテストに分割
        split_ratio = 0.8
        split_index = int(num_data * split_ratio)
        
        X_train, y_train = X[:split_index], y[:split_index]
        X_test, y_test = X[split_index:], y[split_index:]
        
        adj_matrix_train = np.tile(adj_matrix[np.newaxis, :, :], (X_train.shape[0], 1, 1))
        adj_matrix_test = np.tile(adj_matrix[np.newaxis, :, :], (X_test.shape[0], 1, 1))

        return (X_train, y_train, adj_matrix_train), (X_test, y_test, adj_matrix_test), num_links, moved_link_index

# --- 2. モデルの構築（GraphSAGE風） ---

class GraphSAGE(layers.Layer):
    """
    A simplified GraphSAGE-like layer.
    It aggregates features from neighboring nodes.
    """
    def __init__(self, output_dim, **kwargs):
        super(GraphSAGE, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.dense_aggregate = layers.Dense(output_dim, activation='relu')
        self.dense_self = layers.Dense(output_dim, activation='relu')

    def call(self, inputs):
        features, adj_matrix = inputs
        
        # 集約: 隣接ノードの特徴量を合計
        aggregated_features = tf.matmul(adj_matrix, features)
        
        # 結合: 自己特徴量と集約した特徴量を結合
        combined_features = tf.concat([self.dense_self(features), self.dense_aggregate(aggregated_features)], axis=-1)
        
        # 活性化と正規化
        output = tf.nn.relu(combined_features)
        
        return output

def build_model(num_links, input_dim):
    """
    Build the GNN model for rope deformation prediction.
    
    Args:
        num_links (int): The number of links in the rope.
        input_dim (int): The dimension of the input features per node.
    
    Returns:
        keras.Model: The compiled GNN model.
    """
    # 入力層: グラフの特徴量と隣接行列
    node_features_input = keras.Input(shape=(num_links, input_dim), name="node_features_input")
    adj_matrix_input = keras.Input(shape=(num_links, num_links), name="adj_matrix_input")

    # Stack GraphSAGE layers
    x = GraphSAGE(64)([node_features_input, adj_matrix_input])
    x = GraphSAGE(64)([x, adj_matrix_input])
    x = GraphSAGE(64)([x, adj_matrix_input])

    # Output layer: Predict final 3D position
    output = layers.Dense(3)(x)
    
    # Define and compile the model
    model = keras.Model(inputs=[node_features_input, adj_matrix_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# --- 3. メインの学習プロセスと推論 ---

def visualize_combined_results(predictions, y_inference, X_inference, moved_link_index, num_links):
    """
    Visualize the initial state, prediction results, per-node error, and the moved node in a single plot.
    
    Args:
        predictions (np.ndarray): The predicted positions.
        y_inference (np.ndarray): The actual positions (ground truth).
        X_inference (np.ndarray): The initial positions and applied impulses.
        moved_link_index (int): The index of the moved link.
    """
    print("\n--- 推論結果を可視化します ---")
    print("モデルの予測と実際の形状、各ノードの誤差を3Dで表示します。")
    print("赤い大きな点が力を加えたノード、色が誤差の大きさ（緑=低、赤=高）を表します。")
    print("青い紐が力の加える前の形状です。赤い矢印が加えた力を表します。")

    # 各ノードのユークリッド距離誤差を計算
    per_node_error = np.sqrt(np.sum((predictions - y_inference)**2, axis=-1))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sample_idx = random.randint(0, len(predictions) - 1)
    
    # Get the data for the selected sample
    initial_pos = X_inference[sample_idx, :, :3]
    initial_impulse = X_inference[sample_idx, moved_link_index, 3:]
    pred_pos = predictions[sample_idx]
    gt_pos = y_inference[sample_idx]
    errors = per_node_error[sample_idx]
    
    # インデックスが範囲外かチェック
    highlight_node_index = None
    if 0 <= moved_link_index < num_links:
        highlight_node_index = moved_link_index

    # プロットの軸を固定
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    # Plot initial state
    ax.plot(initial_pos[:, 0], initial_pos[:, 1], initial_pos[:, 2], '-', c='cyan', alpha=0.8, label='Initial Rope')
    ax.scatter(initial_pos[:, 0], initial_pos[:, 1], initial_pos[:, 2], c='cyan', s=50)
    ax.quiver(initial_pos[moved_link_index, 0], initial_pos[moved_link_index, 1], initial_pos[moved_link_index, 2],
              initial_impulse[0], initial_impulse[1], initial_impulse[2], 
              color='red', length=0.5, arrow_length_ratio=0.3, label='Applied Impulse')
    
    # 予測された紐を線でプロット
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], '-', c='blue', alpha=0.5, label='Predicted Rope')
    # 正解の紐を線でプロット
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], '--', c='black', alpha=0.5, label='Ground Truth Rope')

    # ノードを散布図でプロット、誤差を色で表現
    scatter = ax.scatter(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 
                         c=errors, cmap='viridis', s=50, label='Error per Node')
    
    # 力を加えたノードを特別な色とサイズでハイライト（有効なインデックスの場合のみ）
    if highlight_node_index is not None:
        ax.scatter(gt_pos[highlight_node_index, 0], gt_pos[highlight_node_index, 1], gt_pos[highlight_node_index, 2],
                   c='red', s=200, label='Moved Node')

    # カラーバーを追加
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Euclidean Distance Error')

    ax.set_title("Rope Deformation Prediction: Initial State vs. Final State")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.legend()
    plt.savefig("/srv/workdir/output.png")

def main():
    # 学習データのロードと前処理
    training_data = load_and_preprocess_data(filename="rope_deformation_data_friction_1.3_normalized.npz", test_data=False)
    if training_data[0] is None:
        return
    (X_train, y_train, adj_matrix_train), (X_test_split, y_test_split, adj_matrix_test_split), num_links, _ = training_data

    # モデルの構築
    model = build_model(num_links, X_train.shape[-1])
    model.summary()

    print("\n--- モデルの学習を開始します ---")
    
    # モデルの学習
    history = model.fit(
        [X_train, adj_matrix_train],
        y_train,
        epochs=500,
        batch_size=100,
        validation_data=([X_test_split, adj_matrix_test_split], y_test_split)
    )

    print("\n--- 学習完了 ---")
    
    # モデルを保存
    model.save('rope_deformation_model_friction_1.3.h5')
    print("モデルが 'rope_deformation_model_friction_1.3.h5' に保存されました。")

     # --- 推論と可視化 ---
    print("\n--- 推論用データのロードを開始します ---")
    inference_data_tuple = load_and_preprocess_data(filename="rope_deformation_data_friction_1.3_test_normalized.npz", test_data=True)
    if inference_data_tuple[0] is None:
        return 
    (X_inference, y_inference, adj_matrix_inference) = inference_data_tuple[0]
    num_links = inference_data_tuple[1]
    moved_link_index = inference_data_tuple[2]

    # 訓練済みモデルをロード
    print("モデルをロードしています...")
    try:
        model = keras.models.load_model('rope_deformation_model_friction_1.3.h5', custom_objects={'GraphSAGE': GraphSAGE})
    except OSError:
        print("エラー: モデルファイル 'rope_deformation_model_friction_1.3.h5' が見つかりません。")
        print("モデルを学習してから再度実行してください。")
        return
    
    # 予測を実行
    print("予測を実行中...")
    predictions = model.predict([X_inference, adj_matrix_inference], verbose=0)
    
    # 予測結果の評価
    # 各データポイント、各ノードの誤差を計算
    per_node_error = np.sqrt(np.sum((predictions - y_inference)**2, axis=-1))
    
    print("\n--- 予測誤差 ---")
    print(f"テストデータでの平均予測誤差: {np.mean(per_node_error):.4f} (メートル)")
    
    # ランダムなサンプルのノードごとの誤差を表示
    sample_idx = random.randint(0, len(per_node_error) - 1)
    print("\nランダムなサンプルにおける各ノードの誤差 (メートル):")
    print(per_node_error[sample_idx])

    # 予測結果を可視化
    visualize_combined_results(predictions, y_inference, X_inference, moved_link_index, num_links)

if __name__ == "__main__":
    main()
