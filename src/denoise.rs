use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::audio::{read_audio, to_model_rate, write_pcm16};
use crate::stft::{chunk_count, istft, stft, FREQ, OVERLAP, SEGMENT};

#[derive(Debug)]
pub struct DenoiseError(pub String);

impl DenoiseError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for DenoiseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DenoiseError {}

pub struct Progress {
    pub cancel: AtomicBool,
    pub permille: AtomicU32,
    pub status: Mutex<String>,
}

impl Progress {
    pub fn new() -> Self {
        Self {
            cancel: AtomicBool::new(false),
            permille: AtomicU32::new(0),
            status: Mutex::new(String::new()),
        }
    }

    pub(crate) fn set_status(&self, text: impl Into<String>) {
        if let Ok(mut status) = self.status.lock() {
            *status = text.into();
        }
    }

    pub(crate) fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self::new()
    }
}

struct ModelIo {
    input: String,
    output: String,
}

pub fn find_model() -> Result<PathBuf, DenoiseError> {
    if let Ok(path) = std::env::var("PEARL_DENOISE_MODEL") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
        return Err(DenoiseError::new(format!(
            "PEARL_DENOISE_MODEL 指向的文件不存在: {}",
            path.display()
        )));
    }
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("model.onnx"));
        }
    }
    candidates.push(PathBuf::from("model.onnx"));
    for path in &candidates {
        if path.is_file() {
            return Ok(path.clone());
        }
    }
    Err(DenoiseError::new(
        "找不到 model.onnx. 请把它放在程序同一目录, 或设置 PEARL_DENOISE_MODEL.",
    ))
}

fn load_io(model: &Path) -> Result<ModelIo, DenoiseError> {
    let path = model.with_extension("json");
    let text = std::fs::read_to_string(&path).map_err(|err| {
        DenoiseError::new(format!("找不到模型配置 {}: {err}", path.display()))
    })?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|err| DenoiseError::new(format!("模型配置无效: {err}")))?;
    let input = value
        .get("input")
        .and_then(|item| item.as_str())
        .unwrap_or("")
        .to_string();
    let output = value
        .get("output")
        .and_then(|item| item.as_str())
        .unwrap_or("")
        .to_string();
    if input.is_empty() || output.is_empty() {
        return Err(DenoiseError::new("模型配置缺少 input 或 output"));
    }
    Ok(ModelIo { input, output })
}

pub fn denoise_file(
    input: &Path,
    output: &Path,
    model: &Path,
    progress: &Progress,
) -> Result<(), DenoiseError> {
    if input == output {
        return Err(DenoiseError::new("输出不能和输入是同一个文件"));
    }
    progress.set_status("读取音频");
    let (channels, sample_rate) = read_audio(input)?;
    if progress.cancelled() {
        return Err(DenoiseError::new("已取消"));
    }
    progress.set_status("重采样到 44.1 kHz");
    let mut channels = to_model_rate(channels, sample_rate)?;
    let expand_stereo = nearly_identical_stereo(&channels);
    if expand_stereo {
        channels.truncate(1);
    }
    progress.set_status("加载模型");
    let io = load_io(model)?;
    let environment = ort::init()
        .with_name("pearl-denoise")
        .build()
        .map_err(|err| DenoiseError::new(format!("无法初始化推理运行时: {err}")))?;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut session = Session::builder(&environment)
        .map_err(|err| DenoiseError::new(format!("无法创建会话: {err}")))?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(|err| DenoiseError::new(format!("无法配置会话: {err}")))?
        .with_intra_threads(threads)
        .map_err(|err| DenoiseError::new(format!("无法配置线程: {err}")))?
        .commit_from_file(model)
        .map_err(|err| DenoiseError::new(format!("无法加载模型: {err}")))?;
    if progress.cancelled() {
        return Err(DenoiseError::new("已取消"));
    }
    let mut denoised = denoise_channels(&channels, &mut session, &io, progress)?;
    if expand_stereo {
        denoised.push(denoised[0].clone());
    }
    limit_peak(&mut denoised);
    if denoised.iter().flatten().any(|sample| !sample.is_finite()) {
        return Err(DenoiseError::new("降噪结果里有非有限采样"));
    }
    if progress.cancelled() {
        return Err(DenoiseError::new("已取消"));
    }
    progress.set_status("写入 WAV");
    let temporary = output.with_file_name(format!(
        "{}.restoring.wav",
        output.file_name().and_then(|name| name.to_str()).unwrap_or("out.wav")
    ));
    write_pcm16(&temporary, &denoised, crate::stft::SAMPLE_RATE)?;
    if progress.cancelled() {
        let _ = std::fs::remove_file(&temporary);
        return Err(DenoiseError::new("已取消"));
    }
    if output.exists() {
        std::fs::remove_file(output).map_err(|err| DenoiseError::new(format!("无法覆盖输出: {err}")))?;
    }
    std::fs::rename(&temporary, output).map_err(|err| DenoiseError::new(format!("无法保存输出: {err}")))?;
    progress.permille.store(1000, Ordering::Relaxed);
    progress.set_status(format!("已写入 {}", output.display()));
    Ok(())
}

fn nearly_identical_stereo(channels: &[Vec<f32>]) -> bool {
    if channels.len() != 2 || channels[0].is_empty() {
        return false;
    }
    let step = (channels[0].len() / 200_000).max(1);
    let mut diff = 0.0f64;
    let mut power = 0.0f64;
    let mut count = 0.0f64;
    let mut index = 0;
    while index < channels[0].len() {
        let left = f64::from(channels[0][index]);
        let right = f64::from(channels[1][index]);
        let delta = left - right;
        diff += delta * delta;
        power += left * left;
        count += 1.0;
        index += step;
    }
    let relative = (diff / count).sqrt() / ((power / count).sqrt() + 1e-12);
    relative < 1e-3
}

fn denoise_channels(
    channels: &[Vec<f32>],
    session: &mut Session,
    io: &ModelIo,
    progress: &Progress,
) -> Result<Vec<Vec<f32>>, DenoiseError> {
    let length = channels.first().map(Vec::len).unwrap_or(0);
    if length == 0 {
        return Err(DenoiseError::new("音频是空的"));
    }
    let count = channels.len();
    let mut denoised = vec![vec![0.0f32; length]; count];
    let total = chunk_count(length).max(1);
    let fade = crossfade_window();
    let left = &fade[..OVERLAP];
    let right = &fade[OVERLAP..];
    let mut pointer = 0;
    let mut index = 0usize;
    while pointer < length {
        if progress.cancelled() {
            return Err(DenoiseError::new("已取消"));
        }
        index += 1;
        progress.set_status(format!("降噪 {index}/{total}"));
        progress
            .permille
            .store(((index - 1) as u32 * 1000) / total as u32, Ordering::Relaxed);
        let last = pointer + SEGMENT >= length;
        let valid = if last { length - pointer } else { SEGMENT };
        let mut segment = vec![vec![0.0f32; SEGMENT]; count];
        for (channel, source) in segment.iter_mut().zip(channels) {
            channel[..valid].copy_from_slice(&source[pointer..pointer + valid]);
        }
        let mut predicted = infer_segment(session, io, &segment)?;
        for channel in &mut predicted {
            if channel.len() < SEGMENT {
                channel.resize(SEGMENT, 0.0);
            } else {
                channel.truncate(SEGMENT);
            }
            apply_fade(channel, left, right, pointer == 0, last);
        }
        for (dest, source) in denoised.iter_mut().zip(&predicted) {
            for (slot, sample) in dest[pointer..pointer + valid].iter_mut().zip(&source[..valid]) {
                *slot += *sample;
            }
        }
        pointer = if last {
            length
        } else {
            pointer + SEGMENT - OVERLAP
        };
        progress
            .permille
            .store((index as u32 * 1000) / total as u32, Ordering::Relaxed);
    }
    Ok(denoised)
}

fn crossfade_window() -> Vec<f32> {
    let n = OVERLAP * 2;
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos())
        .collect()
}

fn apply_fade(channel: &mut [f32], left: &[f32], right: &[f32], is_start: bool, last: bool) {
    if is_start && !last {
        for (sample, gain) in channel[SEGMENT - OVERLAP..].iter_mut().zip(right) {
            *sample *= *gain;
        }
    } else if !is_start && last {
        for (sample, gain) in channel[..OVERLAP].iter_mut().zip(left) {
            *sample *= *gain;
        }
    } else if !is_start {
        for (sample, gain) in channel[..OVERLAP].iter_mut().zip(left) {
            *sample *= *gain;
        }
        for (sample, gain) in channel[SEGMENT - OVERLAP..].iter_mut().zip(right) {
            *sample *= *gain;
        }
    }
}

fn infer_segment(
    session: &mut Session,
    io: &ModelIo,
    channels: &[Vec<f32>],
) -> Result<Vec<Vec<f32>>, DenoiseError> {
    let specs: Vec<_> = channels.iter().map(|channel| stft(channel)).collect();
    let frames = specs.first().map(|spec| spec.frames).unwrap_or(0);
    if frames == 0 || specs.iter().any(|spec| spec.frames != frames) {
        return Err(DenoiseError::new("频谱帧数不一致"));
    }
    let batch = specs.len();
    let mut data = vec![0.0f32; batch * frames * FREQ * 2];
    for (index, spec) in specs.iter().enumerate() {
        let start = index * frames * FREQ * 2;
        data[start..start + spec.data.len()].copy_from_slice(&spec.data);
    }
    let array = Array4::from_shape_vec((batch, frames, FREQ, 2), data)
        .map_err(|err| DenoiseError::new(format!("无法整理频谱: {err}")))?;
    let input = TensorRef::from_array_view(&array)
        .map_err(|err| DenoiseError::new(format!("无法准备模型输入: {err}")))?;
    let outputs = session
        .run(ort::inputs![io.input.as_str() => input])
        .map_err(|err| DenoiseError::new(format!("推理失败: {err}")))?;
    let value = outputs
        .get(io.output.as_str())
        .ok_or_else(|| DenoiseError::new(format!("模型没有输出 {}", io.output)))?;
    let (shape, samples) = value
        .try_extract_tensor::<f32>()
        .map_err(|err| DenoiseError::new(format!("无法读取模型输出: {err}")))?;
    let dims: Vec<usize> = shape.iter().copied().map(|dim| dim as usize).collect();
    if dims.len() != 4 || dims[3] != 2 || dims[2] != FREQ {
        return Err(DenoiseError::new(format!("模型输出形状异常: {dims:?}")));
    }
    let out_batch = dims[0];
    let out_frames = dims[1];
    let owned = samples.to_vec();
    let mut restored = Vec::with_capacity(out_batch);
    for index in 0..out_batch.min(batch) {
        let start = index * out_frames * FREQ * 2;
        let end = start + out_frames * FREQ * 2;
        let spec = crate::stft::Spectrum {
            frames: out_frames,
            data: owned[start..end].to_vec(),
        };
        restored.push(istft(&spec, SEGMENT));
    }
    if restored.len() != batch {
        return Err(DenoiseError::new("模型返回的通道数不对"));
    }
    Ok(restored)
}

fn limit_peak(channels: &mut [Vec<f32>]) {
    let mut peak = 0.0f32;
    for sample in channels.iter().flatten() {
        peak = peak.max(sample.abs());
    }
    if peak > 0.99 {
        let gain = 0.99 / peak;
        for sample in channels.iter_mut().flatten() {
            *sample *= gain;
        }
    }
}
