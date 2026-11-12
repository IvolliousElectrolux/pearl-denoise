# Pearl Denoise

带界面的历史录音降噪程序. 降噪用的是施纳贝尔修复流程里的两级 U-Net (Moliner 与 Valimaki 的 historical-recording 模型), 输出 44.1 kHz 16-bit WAV. 不做后面的加混响.

处理方式和原流程一致: Hamming 窗 STFT, 窗长 2048, 帧移 512, 每段 5 秒, 段间 2048 点汉宁重叠相加. 左右声道几乎相同时只推理一次, 再复制成双声道. 峰值超过 0.99 时压到 0.99.

界面用 gpui-kit. Windows 和 macOS 由 GitHub Actions 编译, 产物里带 `model.onnx` 和 `model.json`, 需要和可执行文件放在同一目录.

权重来自本机施纳贝尔流程正在使用的 checkpoint, 发布在本仓库的 `weights` release, 不进 git. 模型代码来自 Eloi Moliner 的 MIT 许可项目 `denoising-historical-recordings`, 见 `tools/vendor/unet.py`.
