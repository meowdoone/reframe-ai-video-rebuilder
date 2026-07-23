# RE:FRAME — 模块化 AI 视频二次重构程序

程序现在有两种入口：

- `main.py`：ASR → LLM 文案重构 → edge-tts → FFmpeg 画音重构 → ExifTool 元数据的批处理主入口。
- `app.py`：原有的本地网页拆镜、重排与字幕操作台。

## 公开的 AI 内容发布前检查流程

[打开公开流程图](https://meowdoone.github.io/reframe-ai-video-rebuilder/)：从素材来源与授权开始，逐项检查原创贡献、事实、配音、画面、AI/商业披露、音乐许可、技术格式和人工终审，并输出 PASS、REWORK 或 BLOCK。

可编辑的 Mermaid 源文件位于 [`docs/ai-content-check-flow.mmd`](docs/ai-content-check-flow.mmd)。这套流程用于发布前质量与风险控制，不把哈希变化、删除水印或轻微滤镜当作原创证明。

## 模块结构

```text
main.py                 CLI 参数、工作流调度、批量处理、产物清单
asr_module.py           FFmpeg 音频提取、Whisper 转写与默认文本降级
llm_module.py           OpenAI / DeepSeek 兼容的解说词重构
tts_module.py           edge-tts 异步拟人旁白生成
video_renderer.py       0.95 裁切、9:16 标化、色彩调整与彻底换轨
metadata_injector.py    ExifTool 手机参数写入与回读验证
config.py               JSON、环境变量与模块配置
utils.py                命令执行、视频探测、日志、清理与异常
pipeline.py             原网页视频编辑工作流的兼容实现
```

## 安装

需要 Python 3.9+、FFmpeg、ffprobe 和可选的 ExifTool：

```bash
python3 -m pip install -r requirements.txt
```

默认 ASR 使用已缓存的 Whisper `base` 模型；也可以安装 `faster-whisper` 后在配置中切换。

## 单视频运行

```bash
python3 main.py input.mp4 \
  --output-dir output/reconstructed \
  --asr-backend whisper \
  --asr-model base
```

处理结果包括：

- `*_reconstructed.mp4`：1080×1920 H.264/AAC 成片
- `*_transcript.json`：ASR 文本与时间戳
- `*_narration.txt`：重构后的解说词
- `*_manifest.json`：每一步的模式、警告、参数、校验与 SHA256

## 批量处理

输入多个视频或整个目录：

```bash
python3 main.py video-a.mp4 video-b.mp4 ./incoming \
  --recursive \
  --output-dir output/reconstructed
```

单个文件失败不会中断剩余任务；需要遇错即停时加 `--fail-fast`。

## OpenAI / DeepSeek 配置

复制示例配置并填入自己的参数：

```bash
cp config.example.json config.local.json
python3 main.py input.mp4 --config config.local.json
```

也可以使用环境变量，避免把密钥写进文件：

```bash
export VIDEO_LLM_API_KEY="your-key"
export VIDEO_LLM_API_BASE="https://api.deepseek.com/v1"
export VIDEO_LLM_MODEL="deepseek-chat"
python3 main.py input.mp4
```

OpenAI 兼容接口使用 `chat.completions`。如果密钥缺失、失效或接口失败，程序保留 ASR 文本继续执行，并在 manifest 中记录 `source-fallback`。

## ASR 与降级

- `auto`：先尝试 `faster-whisper`，再尝试 `openai-whisper`。
- `whisper`：直接使用本机 `openai-whisper`。
- `faster-whisper`：只使用 faster-whisper。
- Whisper 不可用或转写失败时，使用配置里的 `asr.default_text`。
- 没有默认文本时，ASR/LLM/TTS 标记为跳过，渲染器仍输出彻底移除原声的静音成片。

也可以直接指定默认文案：

```bash
python3 main.py input.mp4 \
  --skip-asr \
  --default-text "前三秒先展示结果，中段说明细节，结尾给出行动提示。"
```

## TTS

默认声音为 `zh-CN-YunxiNeural`：

```bash
python3 main.py input.mp4 --voice zh-CN-YunxiNeural
```

edge-tts 不可用或生成失败时不会回用原视频声音，而是输出静音音轨并记录降级状态。

## 视频渲染

默认执行：

- 中心边缘裁切：`crop_scale=0.95`
- 标准化：1080×1920、9:16、30 FPS
- 色彩：`contrast=1.05`、`saturation=1.08`、`brightness=0.02`
- 完全排除原音，只映射新旁白；没有旁白时使用静音轨
- H.264、AAC、`yuv420p`、`+faststart`
- 视频与旁白使用 `-shortest` 对齐

## 元数据

默认调用 ExifTool 写入并回读：

- Make / Model / Software
- CreateDate / ModifyDate
- TrackCreateDate / TrackModifyDate
- MediaCreateDate / MediaModifyDate

`HandlerDescription` 是 ExifTool 只读字段，因此由 FFmpeg 在封装阶段通过 `handler_name` 写入。未安装 ExifTool 或写入失败时，只打印警告并保留已经生成的视频。

关闭元数据步骤：

```bash
python3 main.py input.mp4 --no-metadata
```

## 网页操作台

```bash
python3 app.py
```

打开 `http://127.0.0.1:8765`，可继续使用多视频拆镜、镜头重排、动态裁切、字幕和背景音乐功能。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖模块降级、配置解析、真实 FFmpeg 换轨渲染、批处理主工作流以及原网页视频编辑流程。
