"""
ComfyUI custom nodes for LTX-2.3 Pro via Replicate API.
Supports: text-to-video, image-to-video, audio-to-video, extend, retake.
Model: lightricks/ltx-2.3-pro
"""

import os
import io
import time
import base64
import tempfile
import subprocess
import requests
import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False

REPLICATE_API_URL = "https://api.replicate.com/v1/models/lightricks/ltx-2.3-pro/predictions"
REPLICATE_POLL_URL = "https://api.replicate.com/v1/predictions/{id}"

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD HELPER — catbox.moe (Replicate requires public HTTPS URLs for image)
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_public(data: bytes, filename: str, mime: str) -> str:
    """Upload bytes to catbox.moe (permanent) or litterbox (72h fallback).
    Returns a public HTTPS URL. Required because Replicate ignores data URIs
    for image inputs in some models.
    """
    # 1. catbox.moe — permanent, no hotlink restrictions
    try:
        r = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload", "userhash": ""},
            files={"fileToUpload": (filename, data, mime)},
            timeout=60,
        )
        r.raise_for_status()
        url = r.text.strip()
        if url.startswith("https://"):
            print(f"[RLTX] Uploaded ({filename}) → {url}")
            return url
    except Exception as e:
        print(f"[RLTX] catbox.moe failed ({e}), trying litterbox...")

    # 2. litterbox — 72h fallback
    try:
        r = requests.post(
            "https://litterbox.catbox.moe/resources/internals/api.php",
            data={"reqtype": "fileupload", "time": "72h"},
            files={"fileToUpload": (filename, data, mime)},
            timeout=60,
        )
        r.raise_for_status()
        url = r.text.strip()
        if url.startswith("https://"):
            print(f"[RLTX] Uploaded (litterbox) → {url}")
            return url
    except Exception as e:
        print(f"[RLTX] litterbox failed ({e}), trying uguu.se...")

    # 3. uguu.se — last resort
    r = requests.post("https://uguu.se/upload",
                      files={"files[]": (filename, data, mime)}, timeout=60)
    r.raise_for_status()
    url = r.json()["files"][0]["url"]
    print(f"[RLTX] Uploaded (uguu.se) → {url}")
    return url


TASKS = ["text_to_video", "image_to_video", "audio_to_video", "extend", "retake"]
RESOLUTIONS = ["1080p", "720p", "480p"]
ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "4:5"]
CAMERA_MOTIONS = ["none", "zoom_in", "zoom_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "rotate_cw", "rotate_ccw", "static"]
EXTEND_MODES = ["end", "beginning"]
RETAKE_MODES = ["replace_audio_and_video", "replace_video_only", "replace_audio_only"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _find_ffmpeg():
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg",
              "/opt/homebrew/bin/ffmpeg", "/usr/local/opt/ffmpeg/bin/ffmpeg",
              "/opt/conda/bin/ffmpeg"]:
        if os.path.exists(p):
            return p
    return None


def tensor_to_jpeg_bytes(image_tensor, max_dim=1920) -> bytes:
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    np_img = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(np_img)
    w, h = pil.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        pil = pil.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def bytes_to_data_uri(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def audio_tensor_to_mp3(audio: dict) -> bytes:
    """Convert ComfyUI AUDIO dict to MP3 bytes. ffmpeg-first pipeline."""
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.shape[0] > 2:
        waveform = waveform[:2]
    channels = waveform.shape[0]

    # 1. ffmpeg PCM → MP3
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        try:
            pcm = (waveform.cpu().numpy() * 32767).clip(-32768, 32767).astype("int16")
            pcm_bytes = pcm.T.flatten().tobytes()
            cmd = [ffmpeg, "-y",
                   "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels),
                   "-i", "pipe:0", "-vn", "-ar", "44100", "-ac", "2",
                   "-b:a", "192k", "-f", "mp3", "pipe:1"]
            res = subprocess.run(cmd, input=pcm_bytes, capture_output=True, timeout=30)
            if res.returncode == 0 and res.stdout:
                return res.stdout
        except Exception as e:
            print(f"[RLTX] ffmpeg failed ({e}), trying torchaudio...")

    # 2. torchaudio MP3
    try:
        import torchaudio
        buf = io.BytesIO()
        torchaudio.save(buf, waveform.cpu(), sample_rate, format="mp3")
        return buf.getvalue()
    except Exception as e:
        print(f"[RLTX] torchaudio MP3 failed ({e}), trying WAV...")

    # 3. torchaudio WAV
    try:
        import torchaudio
        buf = io.BytesIO()
        torchaudio.save(buf, waveform.cpu(), sample_rate, format="wav")
        return buf.getvalue()
    except Exception:
        pass

    # 4. Python wave fallback
    import wave as wavemod
    pcm = (waveform.cpu().numpy() * 32767).clip(-32768, 32767).astype("int16")
    pcm_bytes = pcm.T.flatten().tobytes()
    buf = io.BytesIO()
    with wavemod.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def replicate_run(api_key: str, inputs: dict, poll_interval: float = 3.0, timeout: int = 600) -> str:
    """Submit prediction to Replicate and poll until done. Returns video URL."""
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    }
    r = requests.post(REPLICATE_API_URL, json={"input": inputs}, headers=headers, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"Replicate submit failed {r.status_code}: {r.text[:400]}")
    pred = r.json()
    pred_id = pred["id"]
    print(f"[RLTX] Prediction {pred_id} — status: {pred.get('status')}")

    # Poll
    poll_url = REPLICATE_POLL_URL.format(id=pred_id)
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll_interval)
        pr = requests.get(poll_url, headers={"Authorization": f"Token {api_key}"}, timeout=15)
        pr.raise_for_status()
        data = pr.json()
        status = data.get("status")
        print(f"[RLTX] Status: {status}")
        if status == "succeeded":
            output = data.get("output")
            if isinstance(output, list):
                return output[0]
            return output
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Replicate prediction {status}: {data.get('error', '')}")
    raise RuntimeError(f"Replicate timeout after {timeout}s")


def download_video(url: str) -> bytes:
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    return r.content


def save_video_to_comfy(video_bytes: bytes, prefix: str = "rltx") -> tuple:
    """Save video bytes and return (IMAGE tensor, video_path, ui_dict)."""
    if COMFY_AVAILABLE:
        out_dir = folder_paths.get_output_directory()
    else:
        out_dir = tempfile.gettempdir()
    ts = int(time.time())
    out_path = os.path.join(out_dir, f"{prefix}_{ts}.mp4")
    with open(out_path, "wb") as f:
        f.write(video_bytes)
    print(f"[RLTX] Video saved: {out_path}")

    # Build UI dict for ComfyUI video preview
    if COMFY_AVAILABLE:
        rel = os.path.relpath(out_path, out_dir)
        ui_dict = {"videos": [{"filename": os.path.basename(rel),
                                "subfolder": os.path.dirname(rel),
                                "type": "output"}]}
    else:
        ui_dict = {}

    frames = _decode_video_frames(video_bytes)
    return frames, out_path, ui_dict


def _decode_video_frames(video_bytes: bytes):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes)
    tmp.close()
    try:
        try:
            import torchvision.io as tvio
            frames, _, _ = tvio.read_video(tmp.name, pts_unit="sec", output_format="THWC")
            return frames.float() / 255.0
        except Exception:
            pass
        try:
            import cv2
            cap = cv2.VideoCapture(tmp.name)
            flist = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                flist.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if flist:
                return torch.from_numpy(np.stack(flist)).float() / 255.0
        except Exception:
            pass
        # fallback: single black frame
        return torch.zeros(1, 64, 64, 3)
    finally:
        os.unlink(tmp.name)


# ─────────────────────────────────────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────────────────────────────────────

class RLTXTextToVideo:
    """Generate video from text prompt via Replicate LTX-2.3 Pro."""
    CATEGORY = "LTX Video (Replicate)"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Replicate API token from replicate.com/account"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic scene"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 6, "min": 2, "max": 30, "step": 1}),
                "resolution": (RESOLUTIONS, {"default": "1080p"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "9:16"}),
                "fps": ("INT", {"default": 25, "min": 8, "max": 50}),
                "camera_motion": (CAMERA_MOTIONS, {"default": "none"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }

    def generate(self, api_key, prompt, negative_prompt="", duration=6, resolution="1080p",
                 aspect_ratio="9:16", fps=25, camera_motion="none", generate_audio=True, seed=-1):
        if not api_key.strip():
            raise ValueError("Replicate API key is required.")
        inputs = {
            "task": "text_to_video",
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "fps": fps,
            "camera_motion": camera_motion,
            "generate_audio": generate_audio,
        }
        if negative_prompt:
            inputs["negative_prompt"] = negative_prompt
        if seed >= 0:
            inputs["seed"] = seed

        print(f"[RLTX] Text-to-Video | {aspect_ratio} {resolution} {duration}s")
        url = replicate_run(api_key.strip(), inputs)
        video_bytes = download_video(url)
        frames, path = save_video_to_comfy(video_bytes, "rltx_t2v")
        return (frames, path)


class RLTXImageToVideo:
    """Animate a still image using Replicate LTX-2.3 Pro."""
    CATEGORY = "LTX Video (Replicate)"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Replicate API token"}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "last_frame_image": ("IMAGE",),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "duration": ("INT", {"default": 6, "min": 2, "max": 30}),
                "resolution": (RESOLUTIONS, {"default": "1080p"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "9:16"}),
                "fps": ("INT", {"default": 25, "min": 8, "max": 50}),
                "camera_motion": (CAMERA_MOTIONS, {"default": "none"}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }

    def generate(self, api_key, image, prompt="", last_frame_image=None, negative_prompt="",
                 duration=6, resolution="1080p", aspect_ratio="9:16", fps=25,
                 camera_motion="none", generate_audio=False, seed=-1):
        if not api_key.strip():
            raise ValueError("Replicate API key is required.")

        img_bytes = tensor_to_jpeg_bytes(image)
        inputs = {
            "task": "image_to_video",
            "image": bytes_to_data_uri(img_bytes, "image/jpeg"),
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "fps": fps,
            "camera_motion": camera_motion,
            "generate_audio": generate_audio,
        }
        if last_frame_image is not None:
            lf_bytes = tensor_to_jpeg_bytes(last_frame_image)
            inputs["last_frame_image"] = bytes_to_data_uri(lf_bytes, "image/jpeg")
        if negative_prompt:
            inputs["negative_prompt"] = negative_prompt
        if seed >= 0:
            inputs["seed"] = seed

        print(f"[RLTX] Image-to-Video | {aspect_ratio} {resolution} {duration}s")
        url = replicate_run(api_key.strip(), inputs)
        video_bytes = download_video(url)
        frames, path = save_video_to_comfy(video_bytes, "rltx_i2v")
        return (frames, path)


class RLTXAudioToVideo:
    """Lip-sync video from image + audio using Replicate LTX-2.3 Pro."""
    CATEGORY = "LTX Video (Replicate)"
    FUNCTION = "generate"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "VIDEO", "FLOAT")
    RETURN_NAMES = ("frames", "video", "fps_out")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Replicate API token"}),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "resolution": (RESOLUTIONS, {"default": "1080p"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "9:16"}),
                "fps": ("INT", {"default": 25, "min": 8, "max": 50}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }

    def generate(self, api_key, image, audio, prompt="", resolution="1080p",
                 aspect_ratio="9:16", fps=25, seed=-1):
        if not api_key.strip():
            raise ValueError("Replicate API key is required.")

        print("[RLTX] Encoding + uploading image to catbox.moe...")
        img_bytes = tensor_to_jpeg_bytes(image)
        image_url = upload_to_public(img_bytes, "rltx_image.jpg", "image/jpeg")

        print("[RLTX] Encoding audio → MP3 + uploading...")
        audio_bytes = audio_tensor_to_mp3(audio)
        audio_mime = "audio/mpeg"
        audio_url = upload_to_public(audio_bytes, "rltx_audio.mp3", audio_mime)

        inputs = {
            "task": "audio_to_video",
            "image": image_url,
            "audio": audio_url,
            "prompt": prompt,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "fps": fps,
        }
        if seed >= 0:
            inputs["seed"] = seed

        print(f"[RLTX] Audio-to-Video | {aspect_ratio} {resolution}")
        url = replicate_run(api_key.strip(), inputs)
        video_bytes = download_video(url)
        frames, path, ui = save_video_to_comfy(video_bytes, "rltx_a2v")
        return {"ui": ui, "result": (frames, path, float(fps))}


class RLTXExtendVideo:
    """Extend an existing video using Replicate LTX-2.3 Pro."""
    CATEGORY = "LTX Video (Replicate)"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Replicate API token"}),
                "video_path": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "extend_mode": (EXTEND_MODES, {"default": "end"}),
                "duration": ("INT", {"default": 4, "min": 2, "max": 30}),
                "resolution": (RESOLUTIONS, {"default": "1080p"}),
                "fps": ("INT", {"default": 25, "min": 8, "max": 50}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }

    def generate(self, api_key, video_path, prompt="", extend_mode="end",
                 duration=4, resolution="1080p", fps=25, seed=-1):
        if not api_key.strip():
            raise ValueError("Replicate API key is required.")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = bytes_to_data_uri(video_bytes, "video/mp4")

        inputs = {
            "task": "extend",
            "video": video_b64,
            "prompt": prompt,
            "extend_mode": extend_mode,
            "duration": duration,
            "resolution": resolution,
            "fps": fps,
        }
        if seed >= 0:
            inputs["seed"] = seed

        print(f"[RLTX] Extend Video | mode={extend_mode} +{duration}s")
        url = replicate_run(api_key.strip(), inputs)
        out_bytes = download_video(url)
        frames, path = save_video_to_comfy(out_bytes, "rltx_ext")
        return (frames, path)


class RLTXRetakeVideo:
    """Retake / re-generate a section of a video using Replicate LTX-2.3 Pro."""
    CATEGORY = "LTX Video (Replicate)"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "tooltip": "Replicate API token"}),
                "video_path": ("STRING", {"default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "retake_start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5}),
                "retake_duration": ("FLOAT", {"default": 2.0, "min": 2.0, "max": 30.0, "step": 0.5}),
                "retake_mode": (RETAKE_MODES, {"default": "replace_audio_and_video"}),
                "resolution": (RESOLUTIONS, {"default": "1080p"}),
                "fps": ("INT", {"default": 25, "min": 8, "max": 50}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }

    def generate(self, api_key, video_path, prompt="", retake_start_time=0.0,
                 retake_duration=2.0, retake_mode="replace_audio_and_video",
                 resolution="1080p", fps=25, seed=-1):
        if not api_key.strip():
            raise ValueError("Replicate API key is required.")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = bytes_to_data_uri(video_bytes, "video/mp4")

        inputs = {
            "task": "retake",
            "video": video_b64,
            "prompt": prompt,
            "retake_start_time": retake_start_time,
            "retake_duration": retake_duration,
            "retake_mode": retake_mode,
            "resolution": resolution,
            "fps": fps,
        }
        if seed >= 0:
            inputs["seed"] = seed

        print(f"[RLTX] Retake Video | {retake_start_time}s → +{retake_duration}s | mode={retake_mode}")
        url = replicate_run(api_key.strip(), inputs)
        out_bytes = download_video(url)
        frames, path = save_video_to_comfy(out_bytes, "rltx_ret")
        return (frames, path)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RLTXTextToVideo":   RLTXTextToVideo,
    "RLTXImageToVideo":  RLTXImageToVideo,
    "RLTXAudioToVideo":  RLTXAudioToVideo,
    "RLTXExtendVideo":   RLTXExtendVideo,
    "RLTXRetakeVideo":   RLTXRetakeVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RLTXTextToVideo":   "LTX Text to Video (Replicate) 📝➡️🎬",
    "RLTXImageToVideo":  "LTX Image to Video (Replicate) 🖼️➡️🎬",
    "RLTXAudioToVideo":  "LTX Audio to Video (Replicate) 🎤➡️🎬",
    "RLTXExtendVideo":   "LTX Extend Video (Replicate) ➕🎬",
    "RLTXRetakeVideo":   "LTX Retake Section (Replicate) 🔁🎬",
}

print("[RLTX] LTX-2.3 Pro (Replicate) nodes loaded ✅ — T2V / I2V / A2V / Extend / Retake")
