import numpy as np

# --- 1. データのロード ---
def load_data(filename):
    """
    指定されたnpzファイルをロードし、中身を表示する。
    """
    try:
        loaded_data = np.load(filename, allow_pickle=True)
        print(f"ファイルのロードに成功しました: {filename}")
        for key in loaded_data.files:
            print(f"キー: {key}, データの形状: {loaded_data[key].shape}")
        return loaded_data
    except FileNotFoundError:
        print(f"エラー: ファイル '{filename}' が見つかりません。")
        return None

# --- 2. データの正規化 ---
def normalize_data(loaded_data):
    """
    pre_state と post_state のデータの重心からの相対位置を計算し、正規化する。
    
    正規化を最小-最大スケーリングに変更しました。
    """
    pre_state = loaded_data['pre_state']
    post_state = loaded_data['post_state']
    
    # 各サンプル（データ点）の重心を計算
    pre_state_center_of_mass = np.mean(pre_state, axis=1, keepdims=True)
    post_state_center_of_mass = np.mean(post_state, axis=1, keepdims=True)
    
    # 各ノードの重心からの相対位置を計算
    pre_state_rel = pre_state - pre_state_center_of_mass
    post_state_rel = post_state - post_state_center_of_mass
    
    # 最小-最大スケーリングのために、相対位置の最小値と最大値を計算
    min_val = np.min(pre_state_rel, axis=(0, 1))
    max_val = np.max(pre_state_rel, axis=(0, 1))
    
    # 分母がゼロになるのを避ける
    range_val = max_val - min_val
    range_val[range_val == 0] = 1e-8
    
    print("\n--- 正規化に使用する統計量（Min-Max Scaling） ---")
    print(f"各軸の最小値 (min): {min_val}")
    print(f"各軸の最大値 (max): {max_val}")

    # 相対位置データを最小-最大スケーリングで正規化
    pre_state_norm = 2 * ((pre_state_rel - min_val) / range_val) - 1
    post_state_norm = 2 * ((post_state_rel - min_val) / range_val) - 1
    
    # applied_impulses と rope_links はそのまま使用
    applied_impulses = loaded_data['applied_impulses']
    rope_links = loaded_data['rope_links']
    moved_link_index = loaded_data['moved_link_index']

    return pre_state_norm, post_state_norm, applied_impulses, rope_links, moved_link_index, min_val, max_val

# --- 3. メイン処理 ---
if __name__ == "__main__":
    # 正規化したい元のファイル名
    filename = 'rope_deformation_data_friction_1.3_test.npz'
    
    # データをロード
    data = load_data(filename)
    if data is not None:
        # 正規化を実行
        pre_norm, post_norm, impulses, rope_links, moved_link_index, min_val, max_val = normalize_data(data)
        
        # 正規化されたデータを保存する新しいファイル名
        output_filename = '/srv/workdir/rope_deformation_data_friction_1.3_test_normalized.npz'
        
        # 正規化されたデータと、元のデータの一部（applied_impulses, moved_link_index, rope_links）
        # および正規化に使用した統計量（mean, std）を新しいnpzファイルに保存
        np.savez_compressed(
            output_filename,
            pre_state=pre_norm,
            post_state=post_norm,
            applied_impulses=impulses,
            moved_link_index=moved_link_index,
            rope_links=rope_links,
            pre_state_min=min_val,
            pre_state_max=max_val
        )
        print(f"\n正規化されたデータが '{output_filename}' に保存されました。")
