# RE:FRAME — AI 视频二次重构器

本地运行的短视频重构程序。导入一组视频后，它会自动拆镜、估计画面焦点、生成新的镜头顺序与字幕，并输出 9:16 H.264 成片。

## 启动

```bash
python3 app.py
```

浏览器会自动打开 `http://127.0.0.1:8765`。

## 当前已实现

- 多视频上传与后台任务进度
- OpenCV 镜头变化、运动强度与视觉焦点分析
- OpenAI Responses API 多关键帧视觉规划
- AI 不可用时的本地智能编排回退
- 镜头选段、重排、变速、智能 9:16 裁切
- 中文字幕烧录、原音保留、响度统一
- 可选背景音乐混音
- 1080×1920 成片与 720×1280 快速预览
- 成片、镜头方案、字幕与处理记录下载

## AI 配置

程序读取现有的 `OPENAI_API_KEY`。默认视觉模型为 `gpt-5.6-sol`，可以覆盖：

```bash
export VIDEO_REBUILDER_MODEL="gpt-5.6"
```

没有有效 API 凭据时，程序仍会完成拆镜、重排和渲染，并在结果页标记为 `LOCAL SMART`。

## 命令行生成

```bash
python3 app.py render input-a.mp4 input-b.mp4 \
  --brief "前三秒展示结果，中段证明细节，结尾行动提示" \
  --duration 15 \
  --style fast-cut \
  --output-dir workspace/my-run
```

加 `--preview` 输出 720×1280；加 `--no-ai` 强制使用本地编排。

## 运行依赖

- Python 3.9+
- FFmpeg / ffprobe
- OpenCV、NumPy
- OpenAI Python SDK（仅 AI 视觉规划需要）

```bash
python3 -m pip install -r requirements.txt
```

每次任务保存在 `workspace/jobs/<任务ID>/`，包含：

- `reconstructed_tiktok.mp4`
- `plan.json`
- `manifest.json`
- `captions.ass`

