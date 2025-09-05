import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import random

def setup_simulation():
    """Sets up the PyBullet simulation environment."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

def set_friction(body_id, friction=0.9):
    """
    Sets the lateral friction for a given body.
    Args:
        body_id (int): The unique ID of the body.
        friction (float): The lateral friction coefficient.
    """
    # PyBulletのchangeDynamics関数で摩擦係数を設定します。
    # この例では、横方向の摩擦（lateralFriction）のみを設定します。
    # 必要に応じて、回転摩擦（spinningFriction）や転がり摩擦（rollingFriction）も設定できます。
    p.changeDynamics(body_id, -1, lateralFriction=friction)

def create_environment():
    """Creates the table, rope, and manipulator."""
    # Create a plane for the floor
    p.loadURDF("plane.urdf")

    # Create a table as a rigid body
    table_half_extents = [100.0, 100.0, 0.05]
    table_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.8, 0.6, 0.4, 1])
    table_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
    table_position = [0, 0, 0.5]
    table_id = p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=table_collision_shape,
                                 baseVisualShapeIndex=table_visual_shape,
                                 basePosition=table_position)
    # テーブルの摩擦を設定
    set_friction(table_id, friction=1.3)

    # Create a flexible rope as a series of linked spheres
    rope_start_pos = [0, 0.4, table_position[2] + table_half_extents[2] + 0.05]
    num_links = 30
    link_radius = 0.015
    link_mass = 0.01
    links = []

    for i in range(num_links):
        link_pos = [rope_start_pos[0], rope_start_pos[1] - i * link_radius * 2, rope_start_pos[2]]
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=link_radius, rgbaColor=[0.2, 0.2, 0.8, 1])
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=link_radius)
        link_id = p.createMultiBody(baseMass=link_mass,
                                    baseCollisionShapeIndex=collision_shape,
                                    baseVisualShapeIndex=visual_shape,
                                    basePosition=link_pos)
        links.append(link_id)
        # 各球体の摩擦を設定
        set_friction(link_id, friction=1.3)

    # Create joints to connect the links
    for i in range(num_links - 1):
        p.createConstraint(links[i], -1, links[i+1], -1, p.JOINT_POINT2POINT, [0, 0, 0], [0, link_radius, 0], [0, -link_radius, 0])

    # Create a manipulator arm (e.g., a KUKA LBR IIWA)
    robot_start_pos = [0.5, -0.4, 0.6]
    robot_id = p.loadURDF("franka_panda/panda.urdf", robot_start_pos, useFixedBase=True)
    
    # Pandaモデルにはグリッパーが含まれているため、分離してロードする必要はない
    gripper_id = robot_id
    
    return table_id, links, robot_id, gripper_id

def collect_quasi_static_data(rope_links):
    """
    特定の紐のリンクをランダムに動かし、紐全体の変形データと、加えた力の情報を準静的に収集する。
    Args:
        rope_links (list): 紐を構成する全ての球体のボディIDのリスト。
    Returns:
        dict: 収集されたデータを含む辞書。
    """
    
    pre_deformation_data = []
    post_deformation_data = []
    applied_impulses = []
    moved_link_indices = []

    # 力の強さは一定の微小な値に設定
    force_gain = 1.0
    
    # データを取得するステップ数は固定
    total_steps = 20000
    
    print(f"\n--- 紐の変形データ収集開始 ---")
    
    for i in range(total_steps):
        # 各ステップでターゲットとなる紐のリンクをランダムに選択
        target_link_index = random.randint(0, len(rope_links) - 1)
        target_link_id = rope_links[target_link_index]
        moved_link_indices.append(target_link_index)

        # 力を加える前の紐の状態を記録
        current_pre_shape = []
        for link_id in rope_links:
            pos, _ = p.getBasePositionAndOrientation(link_id)
            current_pre_shape.append(pos)
        pre_deformation_data.append(current_pre_shape)

        # 力を加える時間をランダムに設定（1〜100ステップ）
        apply_force_duration = random.randint(1, 100)

        # ターゲットリンクを動かす方向をランダムに設定
        move_direction = np.random.rand(3) - 0.5  # [-0.5, 0.5]の範囲
        move_direction[2] = 0.0 # Z軸方向への動きは無視
        if np.linalg.norm(move_direction) > 0:
            move_direction /= np.linalg.norm(move_direction) # 正規化
        
        # 加えた力を記録
        force_vector = force_gain * move_direction
        
        # ランダムな時間、力を加える
        for _ in range(apply_force_duration):
            p.applyExternalForce(objectUniqueId=target_link_id,
                                 linkIndex=-1,
                                 forceObj=force_vector,
                                 posObj=p.getBasePositionAndOrientation(target_link_id)[0],
                                 flags=p.WORLD_FRAME)
            p.stepSimulation()
        
        # 力積（impulse）を計算：力 × 時間
        impulse = force_vector * apply_force_duration
        applied_impulses.append(impulse)

        # 力を加えた後の紐の状態を記録
        current_post_shape = []
        for link_id in rope_links:
            pos, _ = p.getBasePositionAndOrientation(link_id)
            current_post_shape.append(pos)
        post_deformation_data.append(current_post_shape)
        
        if i % 100 == 0:
            print(f"データ収集中... {i+1}/{total_steps} ステップ完了")
            print(f"今回のターゲットリンク: {target_link_index}")

    print("--- データ収集完了 ---")
    return {
        'pre_state': np.array(pre_deformation_data),
        'post_state': np.array(post_deformation_data),
        'applied_impulses': np.array(applied_impulses),
        'moved_link_index': np.array(moved_link_indices),
        'rope_links': rope_links
    }

def main():
    setup_simulation()
    table, rope, robot, gripper = create_environment()
    
    # Let the rope settle on the table
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1/240.0)

    # 紐の変形データ、動かしたリンクのインデックス、加えた力の情報を収集
    collected_data = collect_quasi_static_data(rope)

    # データをファイルに保存
    output_filename = "rope_deformation_data_friction_1.3.npz"
    np.savez_compressed(output_filename, 
                        pre_state=collected_data['pre_state'], 
                        post_state=collected_data['post_state'],
                        applied_impulses=collected_data['applied_impulses'],
                        moved_link_index=collected_data['moved_link_index'],
                        rope_links=np.array(collected_data['rope_links']))
    print(f"\nデータが '{output_filename}' に保存されました。")
    print(f"保存された力の加える前の状態のデータの形状: {collected_data['pre_state'].shape}")
    print(f"保存された力の加えた後の状態のデータの形状: {collected_data['post_state'].shape}")
    print(f"保存された加えた力積のデータの形状: {collected_data['applied_impulses'].shape}")
    print(f"保存された動かされたリンクのインデックスの形状: {collected_data['moved_link_index'].shape}")
    print(f"紐の球体IDのリスト: {collected_data['rope_links']}")

    # シミュレーションを終了
    p.disconnect()

if __name__ == "__main__":
    main()
