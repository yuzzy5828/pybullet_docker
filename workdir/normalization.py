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
    """
    pre_state = loaded_data['pre_state']
    post_state = loaded_data['post_state']
    
    # 各サンプル（データ点）の重心を計算
    pre_state_center_of_mass = np.mean(pre_state, axis=1, keepdims=True)
    post_state_center_of_mass = np.mean(post_state, axis=1, keepdims=True)
    
    # 各ノードの重心からの相対位置を計算
    pre_state_rel = pre_state - pre_state_center_of_mass
    post_state_rel = post_state - post_state_center_of_mass
    
    # 相対位置の各軸ごとの平均と標準偏差を計算
    mean = np.mean(pre_state_rel, axis=(0, 1))
    std = np.std(pre_state_rel, axis=(0, 1))
    
    # ゼロ除算を避けるための微小な値を加える
    std[std == 0] = 1e-8
    
    print("\n--- 正規化に使用する統計量 ---")
    print(f"各軸の平均 (mean): {mean}")
    print(f"各軸の標準偏差 (std): {std}")

    # 相対位置データを同じ統計量で標準化
    pre_state_norm = (pre_state_rel - mean) / std
    post_state_norm = (post_state_rel - mean) / std
    
    # applied_impulses と rope_links はそのまま使用
    applied_impulses = loaded_data['applied_impulses']
    rope_links = loaded_data['rope_links']
    moved_link_index = loaded_data['moved_link_index']

    return pre_state_norm, post_state_norm, applied_impulses, rope_links, moved_link_index, mean, std

# --- 3. メイン処理 ---
if __name__ == "__main__":
    # 正規化したい元のファイル名
    filename = 'rope_deformation_data_friction_1.3.npz'
    
    # データをロード
    data = load_data(filename)
    if data is not None:
        # 正規化を実行
        pre_norm, post_norm, impulses, rope_links, moved_link_index, mean, std = normalize_data(data)
        
        # 正規化されたデータを保存する新しいファイル名
        output_filename = 'rope_deformation_data_friction_1.3_normalized.npz'
        
        # 正規化されたデータと、元のデータの一部（applied_impulses, moved_link_index, rope_links）
        # および正規化に使用した統計量（mean, std）を新しいnpzファイルに保存
        np.savez_compressed(
            output_filename,
            pre_state=pre_norm,
            post_state=post_norm,
            applied_impulses=impulses,
            moved_link_index=moved_link_index,
            rope_links=rope_links,
            pre_state_mean=mean,
            pre_state_std=std
        )
        print(f"\n正規化されたデータが '{output_filename}' に保存されました。")
