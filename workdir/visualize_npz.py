import numpy as np

# npzファイルをロード
try:
    data = np.load('/srv/workdir/rope_deformation_data_friction_1.3_normalized.npz', allow_pickle=True)

    # 各データのキーと形状を表示
    print("--- npzファイルの内容 ---")
    for key in data.files:
        print(f"キー: {key}, データの形状: {data[key].shape}")
        if key == 'pre_state':
            print("  （例）最初のステップの紐の形状:\n", data[key][0])
        elif key == 'post_state':
            print("  （例）最初のステップの変形後の紐の形状:\n", data[key][0])
        elif key == 'applied_impulses':
            print("  （例）最初のステップの力積:\n", data[key][0])
        elif key == 'moved_link_index':
            # print(f"  （値）動かされたリンクのインデックス: {data[key]}")
            pass

    # 座標の最大値と最小値を表示
    if 'pre_state' in data.files and 'post_state' in data.files:
        all_positions = np.concatenate((data['pre_state'], data['post_state']), axis=0)
        max_coords = np.max(all_positions, axis=(0, 1))
        min_coords = np.min(all_positions, axis=(0, 1))

        print("\n--- 座標の最大値と最小値 ---")
        print(f"X座標の範囲: [{min_coords[0]:.4f}, {max_coords[0]:.4f}]")
        print(f"Y座標の範囲: [{min_coords[1]:.4f}, {max_coords[1]:.4f}]")
        print(f"Z座標の範囲: [{min_coords[2]:.4f}, {max_coords[2]:.4f}]")
    else:
        print("\n座標データ（'pre_state'または'post_state'）がファイル内に見つかりません。")

except FileNotFoundError:
    print("エラー: 'rope_deformation_data.npz' ファイルが見つかりません。")