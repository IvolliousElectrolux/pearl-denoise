use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::denoise::DenoiseError;
use crate::stft::SAMPLE_RATE;

pub fn read_audio(path: &Path) -> Result<(Vec<Vec<f32>>, u32), DenoiseError> {
    let file = File::open(path).map_err(|err| DenoiseError::new(format!("无法打开音频: {err}")))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, stream, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|err| DenoiseError::new(format!("无法识别音频: {err}")))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| DenoiseError::new("没有音频轨"))?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(SAMPLE_RATE);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|err| DenoiseError::new(format!("无法解码: {err}")))?;

    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(SymphoniaError::IoError(_)) if !channels.is_empty() => break,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(_) if !channels.is_empty() => break,
            Err(err) => return Err(DenoiseError::new(format!("读取音频失败: {err}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(err) => return Err(DenoiseError::new(format!("解码失败: {err}"))),
        };
        if sample_buf.is_none() {
            let spec = *decoded.spec();
            channels = vec![Vec::new(); spec.channels.count()];
            sample_buf = Some(SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
        }
        let buffer = sample_buf.as_mut().expect("sample buffer");
        buffer.copy_interleaved_ref(decoded);
        let count = channels.len();
        for (index, sample) in buffer.samples().iter().enumerate() {
            channels[index % count].push(*sample);
        }
    }
    if channels.is_empty() || channels[0].is_empty() {
        return Err(DenoiseError::new("音频是空的"));
    }
    if channels.len() > 2 {
        channels.truncate(2);
    }
    let length = channels.iter().map(Vec::len).min().unwrap_or(0);
    for channel in &mut channels {
        channel.truncate(length);
    }
    Ok((channels, sample_rate))
}

pub fn to_model_rate(channels: Vec<Vec<f32>>, from: u32) -> Result<Vec<Vec<f32>>, DenoiseError> {
    if from == SAMPLE_RATE {
        return Ok(channels);
    }
    if from == 0 || channels.is_empty() {
        return Err(DenoiseError::new("采样率无效"));
    }
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk = 1024;
    let count = channels.len();
    let mut resampler = SincFixedIn::<f32>::new(
        f64::from(SAMPLE_RATE) / f64::from(from),
        2.0,
        params,
        chunk,
        count,
    )
    .map_err(|err| DenoiseError::new(format!("重采样失败: {err}")))?;
    let mut output = vec![Vec::new(); count];
    let length = channels[0].len();
    let mut offset = 0;
    while offset + chunk <= length {
        let views: Vec<&[f32]> = channels
            .iter()
            .map(|channel| &channel[offset..offset + chunk])
            .collect();
        let processed = resampler
            .process(&views, None)
            .map_err(|err| DenoiseError::new(format!("重采样失败: {err}")))?;
        for (dest, block) in output.iter_mut().zip(processed) {
            dest.extend_from_slice(&block);
        }
        offset += chunk;
    }
    if offset < length {
        let rest: Vec<Vec<f32>> = channels
            .iter()
            .map(|channel| channel[offset..].to_vec())
            .collect();
        let views: Vec<&[f32]> = rest.iter().map(Vec::as_slice).collect();
        let processed = resampler
            .process_partial(Some(&views), None)
            .map_err(|err| DenoiseError::new(format!("重采样失败: {err}")))?;
        for (dest, block) in output.iter_mut().zip(processed) {
            dest.extend_from_slice(&block);
        }
    }
    if output.first().map(Vec::is_empty).unwrap_or(true) {
        return Err(DenoiseError::new("重采样结果是空的"));
    }
    Ok(output)
}

pub fn write_pcm16(path: &Path, channels: &[Vec<f32>]) -> Result<(), DenoiseError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|err| DenoiseError::new(format!("无法创建输出目录: {err}")))?;
        }
    }
    let spec = hound::WavSpec {
        channels: channels.len() as u16,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|err| DenoiseError::new(format!("无法写入 WAV: {err}")))?;
    let length = channels.first().map(Vec::len).unwrap_or(0);
    for index in 0..length {
        for channel in channels {
            let sample = (channel[index].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            writer
                .write_sample(sample)
                .map_err(|err| DenoiseError::new(format!("无法写入 WAV: {err}")))?;
        }
    }
    writer
        .finalize()
        .map_err(|err| DenoiseError::new(format!("无法完成 WAV: {err}")))?;
    Ok(())
}
