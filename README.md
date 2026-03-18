# comfyui-replicate-ltx 🎬

ComfyUI custom nodes for **LTX-2.3 Pro** via [Replicate](https://replicate.com/lightricks/ltx-2.3-pro).

High-fidelity video generation up to 4K at 50 FPS. Text, image, and audio-driven.

## Nodes

| Node | Inputs | Output |
|------|--------|--------|
| 📝 **LTX Text to Video** | prompt | video |
| 🖼️ **LTX Image to Video** | IMAGE + prompt | animated video |
| 🎤 **LTX Audio to Video** | IMAGE + AUDIO + prompt | lip-sync video |
| ➕ **LTX Extend Video** | video_path + prompt | extended video |
| 🔁 **LTX Retake Section** | video_path + start/duration | re-generated section |

## Installation

```bash
cd ~/Documents/ComfyUI/custom_nodes
git clone https://github.com/PauldeLavallaz/comfyui-replicate-ltx
pip install requests pillow
```

## Usage

1. Get your API token from [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
2. Add a node (category: **LTX Video (Replicate)**)
3. Paste your token in the `api_key` field
4. Run

## Key Differences vs `comfyui-ltx-node`

| Feature | comfyui-ltx-node | **comfyui-replicate-ltx** |
|---------|-----------------|--------------------------|
| API | ltx.video (direct) | Replicate |
| Audio upload | 0x0.st temp host | data URI (inline, no host needed) |
| Image upload | uguu.se temp host | data URI (inline) |
| Video upload | — | data URI (inline) |
| Auth | LTX API key | Replicate token |

Audio and images are sent as **base64 data URIs** — no external file hosting required.

## Parameters

### Text to Video
| Param | Default | Notes |
|-------|---------|-------|
| `prompt` | — | Describe the scene |
| `duration` | 6 | Seconds (2–30) |
| `resolution` | 1080p | 1080p / 720p / 480p |
| `aspect_ratio` | 9:16 | 16:9, 9:16, 1:1, 4:5... |
| `fps` | 25 | 8–50 |
| `camera_motion` | none | zoom_in, pan_left, rotate_cw... |
| `generate_audio` | true | AI-generated audio |
| `seed` | -1 | -1 = random |

### Image to Video
Same as above, plus:
- `image` — ComfyUI IMAGE tensor (first frame)
- `last_frame_image` — optional last frame for interpolation

### Audio to Video
- `image` — face/character image
- `audio` — ComfyUI AUDIO (any format, converted to MP3 via ffmpeg)
- Any audio format accepted: mp3, wav, ogg, flac, aac, m4a

### Extend / Retake
- `video_path` — path to existing video (STRING output from previous node)
- Extend: `extend_mode` = end / beginning
- Retake: `retake_start_time`, `retake_duration`, `retake_mode`
