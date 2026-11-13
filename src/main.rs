#![cfg_attr(windows, windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, TryRecvError};
use std::sync::Arc;
use std::time::Duration;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::progress::Progress as ProgressBar;
use gpui_kit::component::scroll::ScrollableElement as _;
use gpui_kit::component::ActiveTheme as _;
use gpui_kit::component::*;
use gpui_kit::*;
use pearl_denoise::{denoise_file, find_model, reduce_hiss_file, DenoiseError, Progress};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Method {
    Schnabel,
    Hiss,
}

struct DenoiseApp {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    method: Method,
    status: String,
    percent: f32,
    busy: bool,
    picker: Option<Task<()>>,
    job: Option<Task<()>>,
    cancel: Option<Arc<Progress>>,
}

impl DenoiseApp {
    fn new() -> Self {
        Self {
            input: None,
            output: None,
            method: Method::Schnabel,
            status: "先选择处理方式, 再选择录音. 两种都不加混响.".into(),
            percent: 0.0,
            busy: false,
            picker: None,
            job: None,
            cancel: None,
        }
    }

    fn pick_input(&mut self, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("选择要降噪的音频".into()),
        });
        self.picker = Some(cx.spawn(async move |this, cx| {
            let picked = match receiver.await {
                Ok(Ok(Some(paths))) => paths.into_iter().next(),
                _ => None,
            };
            if let Some(path) = picked {
                this.update(cx, |this, cx| {
                    this.output = Some(default_output(&path, this.method));
                    this.input = Some(path);
                    this.status = "已选择输入".into();
                    cx.notify();
                })
                .ok();
            }
        }));
    }

    fn pick_output(&mut self, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        let directory = self
            .input
            .as_ref()
            .and_then(|path| path.parent().map(PathBuf::from))
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        let method = self.method;
        let name = self
            .input
            .as_ref()
            .map(|path| {
                default_output(path, method)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("denoised.wav")
                    .to_string()
            })
            .unwrap_or_else(|| "denoised.wav".into());
        let receiver = cx.prompt_for_new_path(&directory, Some(&name));
        self.picker = Some(cx.spawn(async move |this, cx| {
            let picked = match receiver.await {
                Ok(Ok(path)) => path,
                _ => None,
            };
            if let Some(path) = picked {
                this.update(cx, |this, cx| {
                    this.output = Some(ensure_wav(path));
                    this.status = "已选择输出".into();
                    cx.notify();
                })
                .ok();
            }
        }));
    }

    fn start(&mut self, cx: &mut Context<Self>) {
        if self.busy {
            return;
        }
        let Some(input) = self.input.clone() else {
            self.status = "先选择输入音频".into();
            cx.notify();
            return;
        };
        let Some(output) = self.output.clone() else {
            self.status = "先选择输出 WAV".into();
            cx.notify();
            return;
        };
        let method = self.method;
        let model = if method == Method::Schnabel {
            match find_model() {
                Ok(path) => Some(path),
                Err(err) => {
                    self.status = err.to_string();
                    cx.notify();
                    return;
                }
            }
        } else {
            None
        };
        let progress = Arc::new(Progress::new());
        let worker_progress = Arc::clone(&progress);
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let result = match method {
                Method::Schnabel => match model.as_deref() {
                    Some(model) => denoise_file(&input, &output, model, &worker_progress),
                    None => Err(DenoiseError::new("施纳贝尔模式需要模型")),
                },
                Method::Hiss => reduce_hiss_file(&input, &output, &worker_progress),
            };
            let _ = sender.send(result);
        });
        self.busy = true;
        self.percent = 0.0;
        self.status = "正在降噪".into();
        self.cancel = Some(Arc::clone(&progress));
        self.job = Some(cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(200))
                    .await;
                let percent = progress.permille.load(Ordering::Relaxed) as f32 / 10.0;
                let status = progress
                    .status
                    .lock()
                    .ok()
                    .map(|text| text.clone())
                    .unwrap_or_default();
                let finished = receiver.try_recv();
                let stop = !matches!(finished, Err(TryRecvError::Empty));
                this.update(cx, |this, cx| {
                    this.percent = percent;
                    if !status.is_empty() {
                        this.status = status;
                    }
                    if stop {
                        this.busy = false;
                        this.cancel = None;
                        match finished {
                            Ok(Ok(())) => {}
                            Ok(Err(err)) => this.status = err.to_string(),
                            Err(_) => this.status = "降噪线程意外退出".into(),
                        }
                    }
                    cx.notify();
                })
                .ok();
                if stop {
                    break;
                }
            }
        }));
        cx.notify();
    }

    fn set_method(&mut self, method: Method, cx: &mut Context<Self>) {
        if self.busy || self.method == method {
            return;
        }
        self.method = method;
        if let Some(input) = self.input.clone() {
            let previous = self.output.clone();
            let auto = [Method::Schnabel, Method::Hiss]
                .into_iter()
                .map(|mode| default_output(&input, mode))
                .collect::<Vec<_>>();
            if previous.as_ref().is_none_or(|path| auto.iter().any(|item| item == path)) {
                self.output = Some(default_output(&input, method));
            }
        }
        self.status = match method {
            Method::Schnabel => "已选择施纳贝尔 U-Net".into(),
            Method::Hiss => "已选择高频嘶声".into(),
        };
        cx.notify();
    }

    fn cancel_job(&mut self, cx: &mut Context<Self>) {
        if let Some(progress) = &self.cancel {
            progress.cancel.store(true, Ordering::SeqCst);
            self.status = "正在取消, 当前这一段算完后停止".into();
            cx.notify();
        }
    }
}

impl Render for DenoiseApp {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let input = self
            .input
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "还没有输入".into());
        let output = self
            .output
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "还没有输出".into());
        div()
            .size_full()
            .bg(cx.theme().background)
            .text_color(cx.theme().foreground)
            .overflow_y_scrollbar()
            .child(
                v_flex()
                    .size_full()
                    .gap_4()
                    .p_5()
                    .child(
                        v_flex()
                            .gap_1()
                            .child(div().text_xl().child("Pearl Denoise"))
                            .child(
                                div().text_sm().child(match self.method {
                                    Method::Schnabel => "施纳贝尔修复用的两级 U-Net, 44.1 kHz, STFT 2048 / 512, 5 秒一段, 重叠 2048. 不追加混响.",
                                    Method::Hiss => "只处理 6500 Hz 以上. 底噪按 -64 dB 估计, 最多削 12 dB, 不跟踪. 中低频保持原样.",
                                }),
                            ),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .child(self.method_button(
                                "mode-unet",
                                "施纳贝尔 U-Net",
                                Method::Schnabel,
                                cx,
                            ))
                            .child(self.method_button(
                                "mode-hiss",
                                "高频嘶声",
                                Method::Hiss,
                                cx,
                            )),
                    )
                    .child(path_card("输入", &input, cx))
                    .child(path_card("输出", &output, cx))
                    .child(
                        v_flex()
                            .gap_2()
                            .child(
                                h_flex()
                                    .justify_between()
                                    .child(div().text_sm().child(self.status.clone()))
                                    .child(div().text_sm().child(format!("{:.0}%", self.percent))),
                            )
                            .child(ProgressBar::new("denoise").value(self.percent)),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                Button::new("input")
                                    .label("选择音频")
                                    .disabled(self.busy)
                                    .on_click(cx.listener(|this, _, _, cx| this.pick_input(cx))),
                            )
                            .child(
                                Button::new("output")
                                    .label("选择输出")
                                    .disabled(self.busy)
                                    .on_click(cx.listener(|this, _, _, cx| this.pick_output(cx))),
                            )
                            .child(
                                Button::new("start")
                                    .primary()
                                    .label("开始降噪")
                                    .disabled(self.busy || self.input.is_none() || self.output.is_none())
                                    .on_click(cx.listener(|this, _, _, cx| this.start(cx))),
                            )
                            .child(
                                Button::new("cancel")
                                    .label("取消")
                                    .disabled(!self.busy)
                                    .on_click(cx.listener(|this, _, _, cx| this.cancel_job(cx))),
                            ),
                    ),
            )
    }
}

impl DenoiseApp {
    fn method_button(
        &self,
        id: &'static str,
        label: &'static str,
        method: Method,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let mut button = Button::new(id)
            .label(label)
            .disabled(self.busy)
            .on_click(cx.listener(move |this, _, _, cx| this.set_method(method, cx)));
        if self.method == method {
            button = button.primary();
        }
        button
    }
}

fn path_card(title: &str, path: &str, cx: &mut Context<DenoiseApp>) -> impl IntoElement {
    v_flex()
        .gap_1()
        .p_3()
        .w_full()
        .rounded_md()
        .border_1()
        .border_color(cx.theme().border)
        .bg(cx.theme().secondary)
        .child(div().text_sm().child(title.to_string()))
        .child(div().text_sm().child(path.to_string()))
}

fn default_output(input: &std::path::Path, method: Method) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("denoised");
    let tag = match method {
        Method::Schnabel => "denoised",
        Method::Hiss => "hiss",
    };
    let name = format!("{stem}.{tag}.wav");
    match input.parent() {
        Some(parent) => parent.join(name),
        None => PathBuf::from(name),
    }
}

fn ensure_wav(path: PathBuf) -> PathBuf {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
    {
        path
    } else {
        path.with_extension("wav")
    }
}

fn main() {
    gpui_kit::application()
        .with_assets(gpui_kit::assets::Assets)
        .run(|cx| {
            gpui_kit::init(cx);
            let bounds = Bounds {
                origin: point(px(120.), px(80.)),
                size: size(px(760.), px(640.)),
            };
            gpui_kit::open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    window_min_size: Some(size(px(640.), px(480.))),
                    ..Default::default()
                },
                cx,
                |window, cx| {
                    window.set_window_title("Pearl Denoise");
                    cx.new(|_| DenoiseApp::new())
                },
            )
            .expect("无法打开窗口");
        });
}
