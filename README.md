# 图片相似度工作台（桌面版）

## 功能
- 目录扫描相似图，导出 Excel
- 参考图在目录中检索相似图，导出 TXT
- A/B 双目录对比，逐组复核、删除、导出复核报告

## 启动（推荐）
### Windows
双击 `launch_studio.bat`
- 自动创建 `.venv`
- 自动安装依赖（优先清华镜像，失败回退官方）
- 自动启动桌面程序
- 若失败会自动保留日志到 `logs/launch_*.log` 并显示错误摘要

### Linux/macOS
```bash
./launch_studio.sh
```

## 打包（Windows）
双击 `packaging/build_windows.bat`。

## 模型与镜像
- OpenCLIP 模型下载默认优先中国镜像，再回退官方。
- 模型缓存默认在：`~/.image_similarity_workbench/hf_cache`

## 工程结构
- `app/main.py`：应用入口
- `app/ui/main_window.py`：界面层（PySide6）
- `app/services/backend.py`：任务调用桥接
- `app/workers/jobs.py`：线程任务封装
- `app/settings/recent_store.py`：最近路径持久化
- `image_similarity_check_python.py`：核心算法与导出逻辑
