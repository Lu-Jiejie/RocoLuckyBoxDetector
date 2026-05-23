# Gifty Box Detector

游戏抽卡保底计数客户端。实时截屏，多算法并行检测目标盒子，命中后自动录制 GIF，管理保底进度。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动

```bash
python box_detector_client.py
```

### 3. 框选检测区域（推荐）

启动后先点击 **「框选区域」** 按钮：
- 窗口自动最小化，弹出全屏截图
- 鼠标拖拽画框，框住游戏里盒子出现的位置
- 松开鼠标确认，窗口恢复

框选后检测只在框内搜索，速度快且避免误检。

### 4. 开始检测

框选完成后切换到游戏窗口，程序在后台自动检测。命中盒子后：
- 自动录制锁定区域 2.5 秒 GIF
- 卡片显示缩略图，点击 **「放大」** 播放 GIF
- 保底计数自动 +1

点击 **「暂停检测」** 可随时停。

## 界面说明

### 顶部工具栏

| 按钮 | 说明 |
|------|------|
| 框选区域 | 手动框选检测范围 |
| 重置检测框 | 清除锁定，恢复整屏搜索 |
| 暂停检测 | 暂停/继续后台检测 |
| + 添加账号 | 多账号独立计数 |

### 算法参数面板

点击 **「算法参数」** 标题展开。可勾选 SIFT / ORB / BRISK 同时运行，任意一个命中即触发截图。

每个算法四个阈值，修改即时生效：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| matches≥ | 最少匹配点对数 | 5 |
| inliers≥ | 最少几何内点数 | 3 |
| ratio\< | Lowe 比例测试阈值 | 0.80 |
| score≥ | 模板匹配最低分 | 0.20 |

### 保底管理

每账号独立：
- **命中次数**：自动累加或手动修改
- **触发保底**：本轮保底清零，保底次数 +1
- **清除全部**：清空该账号所有记录和截图
- **删除**：单张截图下方的红色按钮，确认后删除

## 命令行

```bash
# 离线自测（验证检测器能识别哪些图）
python box_detector_client.py --self-test

# 指定算法
python box_detector_client.py --self-test --algorithm sift
python box_detector_client.py --self-test --algorithm orb

# 指定模板和数据目录
python box_detector_client.py --self-test --template 标准盒子.png --data-dir data

# 独立屏幕检测脚本（OpenCV 预览窗口，无 GUI）
python screen_detect.py --templates data/

# 截屏 CPU 占用探测
python screenshot_cpu_probe.py --duration 10
```

## 文件存储

| 文件 | 说明 |
|------|------|
| `config.json` | 锁定检测框归一化坐标，框选后自动保存 |
| `client_data/state.json` | 各账号命中记录与保底计数 |
| `client_data/captures/<账号>/` | GIF 和 PNG 缩略图 |

## 打包为 exe

```bash
pip install pyinstaller
pyinstaller --onefile --add-data "标准盒子.png;." --name GiftyBoxDetector box_detector_client.py
```

exe 在 `dist/` 目录，88MB，可直接分发。
