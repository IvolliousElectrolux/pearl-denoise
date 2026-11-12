use std::sync::OnceLock;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub const SAMPLE_RATE: u32 = 44_100;
pub const WIN: usize = 2048;
pub const HOP: usize = 512;
pub const SEGMENT: usize = 44_100 * 5;
pub const OVERLAP: usize = 2048;
pub const FREQ: usize = WIN / 2 + 1;

pub struct Spectrum {
    pub frames: usize,
    pub data: Vec<f32>,
}

struct Windows {
    analysis: Vec<f32>,
    inverse: Vec<f32>,
}

fn windows() -> &'static Windows {
    static WINDOWS: OnceLock<Windows> = OnceLock::new();
    WINDOWS.get_or_init(|| {
        let analysis: Vec<f32> = (0..WIN)
            .map(|i| {
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / WIN as f32).cos()
            })
            .collect();
        let inverse = inverse_window(&analysis, HOP);
        Windows { analysis, inverse }
    })
}

fn inverse_window(forward: &[f32], hop: usize) -> Vec<f32> {
    let n = forward.len();
    let overlaps = n.div_ceil(hop);
    let mut padded = vec![0.0f32; overlaps * hop];
    for (i, sample) in forward.iter().enumerate() {
        padded[i] = sample * sample;
    }
    let mut folded = vec![0.0f32; hop];
    for lap in 0..overlaps {
        for i in 0..hop {
            folded[i] += padded[lap * hop + i];
        }
    }
    (0..n)
        .map(|i| {
            let denom = folded[i % hop];
            if denom == 0.0 {
                0.0
            } else {
                forward[i] / denom
            }
        })
        .collect()
}

pub fn stft(samples: &[f32]) -> Spectrum {
    let windows = windows();
    let frames = if samples.is_empty() {
        0
    } else {
        samples.len().div_ceil(HOP)
    };
    let mut data = vec![0.0f32; frames * FREQ * 2];
    if frames == 0 {
        return Spectrum { frames, data };
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(WIN);
    let mut buf = vec![Complex::new(0.0, 0.0); WIN];
    for frame in 0..frames {
        let start = frame * HOP;
        for i in 0..WIN {
            let sample = samples.get(start + i).copied().unwrap_or(0.0);
            buf[i] = Complex::new(sample * windows.analysis[i], 0.0);
        }
        fft.process(&mut buf);
        let offset = frame * FREQ * 2;
        for bin in 0..FREQ {
            data[offset + bin * 2] = buf[bin].re;
            data[offset + bin * 2 + 1] = buf[bin].im;
        }
    }
    Spectrum { frames, data }
}

pub fn istft(spec: &Spectrum, length: usize) -> Vec<f32> {
    let windows = windows();
    if spec.frames == 0 {
        return vec![0.0; length];
    }
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(WIN);
    let out_len = (spec.frames - 1) * HOP + WIN;
    let mut acc = vec![0.0f32; out_len];
    let mut buf = vec![Complex::new(0.0, 0.0); WIN];
    let scale = 1.0 / WIN as f32;
    for frame in 0..spec.frames {
        buf.fill(Complex::new(0.0, 0.0));
        let offset = frame * FREQ * 2;
        buf[0] = Complex::new(spec.data[offset], 0.0);
        for bin in 1..FREQ - 1 {
            let re = spec.data[offset + bin * 2];
            let im = spec.data[offset + bin * 2 + 1];
            buf[bin] = Complex::new(re, im);
            buf[WIN - bin] = Complex::new(re, -im);
        }
        buf[FREQ - 1] = Complex::new(spec.data[offset + (FREQ - 1) * 2], 0.0);
        ifft.process(&mut buf);
        let start = frame * HOP;
        for i in 0..WIN {
            acc[start + i] += buf[i].re * scale * windows.inverse[i];
        }
    }
    if acc.len() >= length {
        acc.truncate(length);
        acc
    } else {
        acc.resize(length, 0.0);
        acc
    }
}

pub fn chunk_count(length: usize) -> usize {
    if length == 0 {
        return 0;
    }
    let mut pointer = 0;
    let mut count = 0;
    while pointer < length {
        count += 1;
        let last = pointer + SEGMENT >= length;
        pointer = if last {
            length
        } else {
            pointer + SEGMENT - OVERLAP
        };
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_hamming_matches_tensorflow() {
        let window = &windows().analysis;
        assert!((window[0] - 0.08).abs() < 1e-6);
        assert!((window[WIN / 2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn stft_roundtrip_keeps_the_sine() {
        let n = SAMPLE_RATE as usize;
        let mut input = vec![0.0f32; n];
        for (i, sample) in input.iter_mut().enumerate() {
            *sample = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin() * 0.25;
        }
        let output = istft(&stft(&input), n);
        let mut err = 0.0f32;
        for i in 4096..n - 4096 {
            err = err.max((input[i] - output[i]).abs());
        }
        assert!(err < 1e-3, "max error {err}");
    }

    #[test]
    fn twelve_seconds_are_three_chunks() {
        assert_eq!(chunk_count(SAMPLE_RATE as usize * 12), 3);
        assert_eq!(chunk_count(100), 1);
    }
}
