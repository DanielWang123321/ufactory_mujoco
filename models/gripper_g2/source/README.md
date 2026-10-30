# Gripper G2 资产快照

`gripper_g2/` 保存 `ufactory_genesis` 工作区的 7 个分体 GLB、7 个碰撞 STL 和 URDF。该工作区基于提交 `4012a34b2af40a51365dbcd2d793a4b2c77971b1`，部分文件当时尚未提交；各输入文件的 SHA-256 见 `gripper_g2/snapshot.json`。`reference_opening_axial_length.csv` 是参考项目验证脚本使用的 85 组 SolidWorks 开度和轴向长度，哈希同样记录在快照清单中。

视觉 OBJ 和碰撞 STL 可从快照重新生成（需要 `py313`、Genesis、trimesh 及 Draco 解码依赖）：

```bash
python scripts/convert_glb_to_obj.py --target gripper_g2 --ref-dir models/gripper_g2/source
```

参考坐标以法兰安装面为 Z=0，外指节 Z=0.0606 m、内指节 Z=0.075602 m、TCP Z=0.1735 m。动力学与开度映射保持项目已有的标定值。
