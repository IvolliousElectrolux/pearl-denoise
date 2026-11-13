# Pearl Denoise

带界面的降噪程序, 有两种处理, 都不加混响.

施纳贝尔模式用修复流程里的两级 U-Net (Moliner 与 Valimaki 的 historical-recording 模型), 输出 44.1 kHz 16-bit WAV.

高频嘶声模式不加载模型. 在 6500 Hz 做四阶 Linkwitz-Riley 分频, 只对高频做固定底噪抑制, 底噪 -64 dB, 最多削 12 dB, 不随音乐跟踪. 中低频原样加回去. 输出保持原来的采样率.

施纳贝尔模式和原流程一致: Hamming 窗 STFT, 窗长 2048, 帧移 512, 每段 5 秒, 段间 2048 点汉宁重叠相加. 左右声道几乎相同时只推理一次, 再复制成双声道. 峰值超过 0.99 时压到 0.99.

界面用 gpui-kit. Windows 和 macOS 由 GitHub Actions 编译, 产物里带 `model.onnx` 和 `model.json`, 需要和可执行文件放在同一目录.

权重来自本机施纳贝尔流程正在使用的 checkpoint, 发布在本仓库的 `weights` release, 不进 git. 模型代码来自 Eloi Moliner 的 MIT 许可项目 `denoising-historical-recordings`, 见 `tools/vendor/unet.py`.
