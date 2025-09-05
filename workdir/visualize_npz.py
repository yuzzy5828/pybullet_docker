import numpy as np

# npzファイルをロード
try:
    data = np.load('rope_deformation_data.npz', allow_pickle=True)

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
            print(f"  （値）動かされたリンクのインデックス: {data[key].item()}")

except FileNotFoundError:
    print("エラー: 'rope_deformation_data.npz' ファイルが見つかりません。")