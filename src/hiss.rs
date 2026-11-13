use std::path::Path;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::audio::{read_audio, write_pcm16};
use crate::denoise::{DenoiseError, Progress};

const SPLIT_HZ: f32 = 6500.0;
const NOISE_FLOOR_DB: f32 = -64.0;
const MAX_REDUCTION_DB: f32 = 12.0;
const GAIN_SMOOTH: usize = 12;

pub fn reduce_hiss_file(
    input: &Path,
    output: &Path,
    progress: &Progress,
) -> Result<(), DenoiseError> {
    if input == output {
        return Err(DenoiseError::new("输出不能和输入是同一个文件"));
    }
    progress.set_status("读取音频");
    let (mut channels, sample_rate) = read_audio(input)?;
    if channels.is_empty() || channels[0].is_empty() {
        return Err(DenoiseError::new("音频是空的"));
    }
    if sample_rate < 16_000 {
        return Err(DenoiseError::new("采样率太低, 无法在 6500 Hz 处分频"));
    }
    let count = channels.len();
    for (index, channel) in channels.iter_mut().enumerate() {
        if progress.cancelled() {
            return Err(DenoiseError::new("已取消"));
        }
        progress.set_status(format!("削高频嘶声 {}/{count}", index + 1));
        *channel = reduce_channel(channel, sample_rate, progress)?;
        let done = ((index + 1) as f32 / count as f32 * 1000.0) as u32;
        progress.permille.store(done.min(999), std::sync::atomic::Ordering::Relaxed);
    }
    if channels.iter().flatten().any(|sample| !sample.is_finite()) {
        return Err(DenoiseError::new("结果里有非有限采样"));
    }
    if progress.cancelled() {
        return Err(DenoiseError::new("已取消"));
    }
    progress.set_status("写入 WAV");
    let temporary = output.with_file_name(format!(
        "{}.restoring.wav",
        output
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("out.wav")
    ));
    write_pcm16(&temporary, &channels, sample_rate)?;
    if progress.cancelled() {
        let _ = std::fs::remove_file(&temporary);
        return Err(DenoiseError::new("已取消"));
    }
    if output.exists() {
        std::fs::remove_file(output)
            .map_err(|err| DenoiseError::new(format!("无法覆盖输出: {err}")))?;
    }
    std::fs::rename(&temporary, output)
        .map_err(|err| DenoiseError::new(format!("无法保存输出: {err}")))?;
    progress
        .permille
        .store(1000, std::sync::atomic::Ordering::Relaxed);
    progress.set_status(format!("已写入 {}", output.display()));
    Ok(())
}

fn reduce_channel(input: &[f32], sample_rate: u32, progress: &Progress) -> Result<Vec<f32>, DenoiseError> {
    let mut low = [lowpass(SPLIT_HZ, sample_rate), lowpass(SPLIT_HZ, sample_rate)];
    let mut high = [highpass(SPLIT_HZ, sample_rate), highpass(SPLIT_HZ, sample_rate)];
    let mut lows = Vec::with_capacity(input.len());
    let mut highs = Vec::with_capacity(input.len());
    for sample in input {
        lows.push(run_pair(&mut low, *sample));
        highs.push(run_pair(&mut high, *sample));
    }
    let highs = suppress_floor(&highs, sample_rate, progress)?;
    Ok(lows
        .into_iter()
        .zip(highs)
        .map(|(low, high)| low + high)
        .collect())
}

fn suppress_floor(
    samples: &[f32],
    sample_rate: u32,
    progress: &Progress,
) -> Result<Vec<f32>, DenoiseError> {
    let hop = (sample_rate / 80).max(1) as usize;
    let window_len = 3 * hop;
    let fft_len = window_len.next_power_of_two();
    let bins = fft_len / 2 + 1;
    let window: Vec<f32> = (0..window_len)
        .map(|i| {
            let raised = (std::f32::consts::PI * i as f32 / window_len as f32).sin();
            raised * raised
        })
        .collect();
    let coherent = window.iter().sum::<f32>() / 2.0;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_len);
    let ifft = planner.plan_fft_inverse(fft_len);
    let scale = 1.0 / fft_len as f32;
    let mut output = vec![0.0f32; samples.len()];
    let mut weight = vec![0.0f32; samples.len()];
    let mut start = 0isize;
    let mut frames = 0u32;
    let total = (samples.len() / hop).max(1) as u32;
    while start < samples.len() as isize {
        if frames.is_multiple_of(32) {
            if progress.cancelled() {
                return Err(DenoiseError::new("已取消"));
            }
            let frac = frames.min(total) as f32 / total as f32;
            progress.permille.store(
                (frac * 1000.0) as u32,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let mut buf = vec![Complex::new(0.0, 0.0); fft_len];
        for (i, tap) in window.iter().enumerate() {
            let index = start + i as isize;
            if index >= 0 && (index as usize) < samples.len() {
                buf[i] = Complex::new(samples[index as usize] * tap, 0.0);
            }
        }
        fft.process(&mut buf);
        let mut gain = vec![1.0f32; bins];
        for (bin, slot) in gain.iter_mut().enumerate() {
            let magnitude = buf[bin].norm();
            let db = 20.0 * (magnitude / coherent).max(1e-12).log10();
            let excess = db - NOISE_FLOOR_DB;
            let reduction = if excess >= MAX_REDUCTION_DB {
                0.0
            } else if excess <= 0.0 {
                MAX_REDUCTION_DB
            } else {
                MAX_REDUCTION_DB * (1.0 - excess / MAX_REDUCTION_DB)
            };
            *slot = 10f32.powf(-reduction / 20.0);
        }
        smooth_gain(&mut gain, GAIN_SMOOTH);
        for bin in 0..bins {
            buf[bin] *= gain[bin];
        }
        for bin in 1..bins - 1 {
            buf[fft_len - bin] *= gain[bin];
        }
        ifft.process(&mut buf);
        for (i, tap) in window.iter().enumerate() {
            let index = start + i as isize;
            if index >= 0 && (index as usize) < samples.len() {
                let at = index as usize;
                output[at] += buf[i].re * scale * tap;
                weight[at] += tap * tap;
            }
        }
        start += hop as isize;
        frames += 1;
    }
    for (sample, tap) in output.iter_mut().zip(weight) {
        if tap > 1e-8 {
            *sample /= tap;
        }
    }
    Ok(output)
}

fn smooth_gain(gain: &mut [f32], radius: usize) {
    if radius == 0 || gain.len() <= radius * 2 {
        return;
    }
    let original = gain.to_vec();
    for i in radius..gain.len() - radius {
        let center = original[i];
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for neighbor in &original[i - radius..=i + radius] {
            let weight = (1.0 - (neighbor - center).abs()).max(0.0);
            num += neighbor * weight;
            den += weight;
        }
        if den > 0.0 {
            gain[i] = num / den;
        }
    }
}

struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

fn lowpass(freq: f32, sample_rate: u32) -> Biquad {
    butterworth(freq, sample_rate, false)
}

fn highpass(freq: f32, sample_rate: u32) -> Biquad {
    butterworth(freq, sample_rate, true)
}

fn butterworth(freq: f32, sample_rate: u32, high: bool) -> Biquad {
    let w0 = 2.0 * std::f32::consts::PI * freq / sample_rate as f32;
    let alpha = w0.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
    let cos = w0.cos();
    let (b0, b1, b2) = if high {
        ((1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0)
    } else {
        ((1.0 - cos) / 2.0, 1.0 - cos, (1.0 - cos) / 2.0)
    };
    let a0 = 1.0 + alpha;
    Biquad {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: (-2.0 * cos) / a0,
        a2: (1.0 - alpha) / a0,
        z1: 0.0,
        z2: 0.0,
    }
}

fn run_pair(pair: &mut [Biquad; 2], sample: f32) -> f32 {
    let mid = pair[0].tick(sample);
    pair[1].tick(mid)
}

impl Biquad {
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, db: f32, n: usize, sample_rate: u32) -> Vec<f32> {
        let amp = 10f32.powf(db / 20.0);
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        let energy = samples.iter().map(|sample| sample * sample).sum::<f32>();
        (energy / samples.len() as f32).sqrt()
    }

    fn db(samples: &[f32]) -> f32 {
        20.0 * rms(samples).max(1e-12).log10()
    }

    #[test]
    fn low_note_stays() {
        let progress = Progress::new();
        let input = sine(200.0, -12.0, 44_100, 44_100);
        let output = reduce_channel(&input, 44_100, &progress).unwrap();
        let skip = 8_000;
        let delta = db(&output[skip..output.len() - skip]) - db(&input[skip..input.len() - skip]);
        assert!(delta.abs() < 0.4, "low note moved by {delta} dB");
    }

    #[test]
    fn loud_highs_stay() {
        let progress = Progress::new();
        let input = sine(14_000.0, -20.0, 44_100, 44_100);
        let output = reduce_channel(&input, 44_100, &progress).unwrap();
        let skip = 8_000;
        let delta = db(&output[skip..output.len() - skip]) - db(&input[skip..input.len() - skip]);
        assert!(delta.abs() < 1.5, "loud highs moved by {delta} dB");
    }

    #[test]
    fn quiet_hiss_drops() {
        let progress = Progress::new();
        let input = sine(14_000.0, -70.0, 44_100, 44_100);
        let output = reduce_channel(&input, 44_100, &progress).unwrap();
        let skip = 8_000;
        let delta = db(&input[skip..input.len() - skip]) - db(&output[skip..output.len() - skip]);
        assert!(
            (8.0..14.0).contains(&delta),
            "quiet highs dropped by {delta} dB"
        );
    }
}
