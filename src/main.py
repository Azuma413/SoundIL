import genesis as gs
# gs.init(backend=gs.cpu) # CPU backend
gs.init(backend=gs.cuda) # GPU backend

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.URDF(
        file = 'URDF/SO_5DOF_ARM100_05d.SLDASM/urdf/SO_5DOF_ARM100_05d.SLDASM.urdf'
    )
)

cam = scene.add_camera(
    res    = (1280, 960),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False
)

scene.build()

# カメラ録画を開始します。開始後、レンダリングされたすべてのRGB画像は内部的に録画されます。
cam.start_recording()

import numpy as np
for i in range(120):
    scene.step()

    # カメラの位置を変更
    cam.set_pose(
        pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat = (0, 0, 0.5),
    )
    
    cam.render()

# 録画を停止してビデオを保存します。`filename`を指定しない場合、呼び出し元のファイル名を使用して名前が自動生成されます。
cam.stop_recording(save_to_filename='video.mp4', fps=60)