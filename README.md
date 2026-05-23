# Gifty Box Detector

洛克王国世界 s2 保底计数客户端 — 基于 OpenCV 特征匹配的实时屏幕目标检测桌面应用。

实时截屏，多算法（SIFT/ORB/BRISK）并行检测目标盒子，命中后自动录制 2.5 秒 GIF 保存。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
# 启动 GUI
python box_detector_client.py

# 命令行自测（离线验证检测器）
python box_detector_client.py --self-test

# 独立屏幕检测脚本（OpenCV 预览窗口）
python screen_detect.py --templates data/

# 截屏 CPU 探测
python screenshot_cpu_probe.py --duration 10
```

## 自测

```bash
# 指定算法
python box_detector_client.py --self-test --algorithm sift
python box_detector_client.py --self-test --algorithm orb

# 自定义模板
python box_detector_client.py --self-test --template 标准盒子.png --data-dir data
```

## 打包

```bash
pip install pyinstaller
pyinstaller --onefile --add-data "标准盒子.png;." box_detector_client.py
```

## 功能

- 多算法并行检测（SIFT / ORB / BRISK 勾选）
- 可调阈值参数面板
- 手动框选检测区域
- 命中后录制 2.5 秒 ROI GIF
- 保底计数（手动编辑 + 触发保底）
- 框选区域锁定 + config.json 持久化

## 存储

- `config.json` — 锁定检测框坐标
- `client_data/state.json` — 各账号命中记录与保底计数
- `client_data/captures/<account_id>/` — GIF 和缩略图
