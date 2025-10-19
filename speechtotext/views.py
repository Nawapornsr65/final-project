# speechtotext/views.py
import os
import time
import tempfile

import torch
import librosa
import soundfile as sf

from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from transformers import pipeline as hf_pipeline


# ───────────────────────────────────────────────────────────
# 1) CONFIG จาก .env (settings.py โหลด dotenv ไปแล้ว)
# ───────────────────────────────────────────────────────────
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "biodatlab/whisper-th-medium-combined")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "30"))

_ENABLE_DIAR = os.getenv("ENABLE_DIARIZATION", "0")
ENABLE_DIARIZATION = _ENABLE_DIAR in {"1", "true", "True", "yes", "YES"}

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

# device: auto|cuda|cpu
_DEVICE_ENV = (os.getenv("DEVICE") or "auto").lower()
if _DEVICE_ENV == "cuda":
    DEVICE = 0 if torch.cuda.is_available() else -1
elif _DEVICE_ENV == "cpu":
    DEVICE = -1
else:
    DEVICE = 0 if torch.cuda.is_available() else -1  # auto

# dtype: ใช้ float16 บน CUDA เพื่อความไว
if DEVICE == -1:
    TORCH_DTYPE = torch.float32
else:
    TORCH_DTYPE = torch.float16

# ───────────────────────────────────────────────────────────
# 2) สร้าง ASR Pipeline (Pathumma/Whisper ไทย)
#    หมายเหตุ: ใช้ generation_config ในการบอก task/language
# ───────────────────────────────────────────────────────────
asr = hf_pipeline(
    task="automatic-speech-recognition",
    model=HF_MODEL_ID,
    device=DEVICE,
    chunk_length_s=CHUNK_SECONDS,
    return_timestamps=True,
    torch_dtype=TORCH_DTYPE if DEVICE != -1 else None,
    generate_kwargs={  # ไม่ใช้ forced_decoder_ids เพื่อตัด warning/ขัดกัน
        "task": "transcribe",
        "language": "th",
    },
)

# ───────────────────────────────────────────────────────────
# 3) เตรียม Pyannote (ใช้ waveform dict → เลี่ยง torchcodec/ffmpeg DLL)
# ───────────────────────────────────────────────────────────
os.environ.setdefault("PYANNOTE_AUDIO_PREFERRED_BACKENDS", "soundfile,torchaudio")
# กันไม่ให้บังคับโหลด torchcodec
os.environ.setdefault("PYANNOTE_AUDIO_DISABLE_TORCHCODEC", "1")

diar_pipeline = None
diar_load_error = None
if ENABLE_DIARIZATION and HF_TOKEN:
    try:
        from pyannote.audio import Pipeline  # type: ignore
        diar_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
        if DEVICE != -1:  # GPU พร้อม
            diar_pipeline.to(torch.device("cuda"))
    except Exception as _e:
        diar_pipeline = None
        diar_load_error = str(_e)


def index(request):
    return render(request, "speechtotext/index.html")


# ───────────────────────────────────────────────────────────
# Helper: ดึง segments ที่มี timestamp จากผลลัพธ์ ASR
# ───────────────────────────────────────────────────────────
def extract_asr_segments(asr_result):
    """
    คืน list ของ dict: {start, end, text}
    รองรับรูปแบบผลลัพธ์ของ transformers whisper pipeline
    """
    chunks = asr_result.get("chunks") or asr_result.get("segments") or []
    out = []
    for c in chunks:
        ts = c.get("timestamp") or [c.get("start"), c.get("end")]
        if ts and len(ts) == 2:
            start = 0.0 if ts[0] is None else float(ts[0])
            end = 0.0 if ts[1] is None else float(ts[1])
            text = (c.get("text") or "").strip()
            if text:
                out.append({"start": start, "end": end, "text": text})
    return out


# ───────────────────────────────────────────────────────────
# Helper: รัน diarization ด้วย waveform dict (ไม่ใช้ไฟล์/torchcodec)
# ───────────────────────────────────────────────────────────
def diarize_segments_from_waveform(y, sr):
    """
    รับ numpy array y (mono) และ sample rate sr
    คืน list ของ {'start', 'end', 'speaker'}
    """
    if diar_pipeline is None:
        return []

    # ให้เป็นเทนเซอร์ (channels, time)
    if y.ndim == 1:
        y_t = torch.from_numpy(y).float().unsqueeze(0)  # (1, T)
    elif y.ndim == 2 and y.shape[0] <= y.shape[1]:
        y_t = torch.from_numpy(y).float()
    else:
        y_t = torch.from_numpy(y.squeeze()).float().unsqueeze(0)

    ann = diar_pipeline({"waveform": y_t, "sample_rate": sr})
    out = []
    for turn, _, spk in ann.itertracks(yield_label=True):
        spk_num = 1
        if isinstance(spk, str) and spk.startswith("SPEAKER_"):
            try:
                spk_num = int(spk.split("_")[-1]) + 1
            except Exception:
                spk_num = 1
        out.append({"start": float(turn.start), "end": float(turn.end), "speaker": spk_num})
    return out


# ───────────────────────────────────────────────────────────
# Helper: จับคู่ speaker ให้ ASR segment ด้วย max-overlap
# ───────────────────────────────────────────────────────────
def assign_speakers(asr_segments, diar_segments):
    if not diar_segments:
        return [{"speaker": None, **a} for a in asr_segments]

    merged = []
    for a in asr_segments:
        best, overlap = None, 0.0
        for d in diar_segments:
            ov = max(0.0, min(a["end"], d["end"]) - max(a["start"], d["start"]))
            if ov > overlap:
                overlap, best = ov, d
        merged.append({"speaker": best["speaker"] if best else None, **a})
    return merged


# ───────────────────────────────────────────────────────────
# View
# ───────────────────────────────────────────────────────────
@csrf_exempt
@require_POST
def speech_to_text(request):
    f = request.FILES.get("file")
    if not f:
        return HttpResponseBadRequest("Missing file (field name: file)")

    t0 = time.time()
    tmp_in = tmp_wav = None

    try:
        # 1) เขียนไฟล์อัปโหลดเป็น temp พร้อมนามสกุลเดิม (ถ้ามี)
        name = getattr(f, "name", "") or "audio.m4a"
        ext = os.path.splitext(name)[1] or ".m4a"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as t:
            tmp_in = t.name
            for chunk in f.chunks():
                t.write(chunk)

        # 2) โหลดเสียง → mono 16k (numpy) ด้วย librosa (เลี่ยง backend แปลกๆ)
        #    ถ้า PySoundFile ใช้ไม่ได้ librosa จะ fallback audioread ให้อัตโนมัติ
        y, sr = librosa.load(tmp_in, sr=16000, mono=True)

        # 3) เขียน WAV 16k เพื่อให้ whisper อ่านเร็ว/ง่าย
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(tmp_wav, y, 16000, subtype="PCM_16")

        # 4) รัน ASR
        asr_result = asr(
            tmp_wav,
            return_timestamps=True,       # ต้องมีเพื่อดึงเวลา
        )
        asr_segments = extract_asr_segments(asr_result)

        # 5) รัน Diarization ด้วย waveform dict (ถ้าเปิดใช้ & pipeline พร้อม)
        diar_segments = []
        diar_used = False
        diar_error = None
        if ENABLE_DIARIZATION and diar_pipeline is not None:
            try:
                diar_segments = diarize_segments_from_waveform(y, 16000)
                diar_used = True
            except Exception as e:
                diar_error = str(e)
                diar_segments = []
                diar_used = False

        merged = assign_speakers(asr_segments, diar_segments)

        # 6) สรุปเป็นข้อความ (ผู้พูดคนที่ N: …)
        lines = []
        for seg in merged:
            spk = f"ผู้พูดคนที่ {seg['speaker']}: " if seg["speaker"] else ""
            lines.append(f"{spk}{seg['text']}")
        full_text = "\n".join(lines)

        t_total = time.time() - t0

        debug = {
            "device_used": "cuda" if DEVICE != -1 else "cpu",
            "torch_dtype": "float16" if TORCH_DTYPE == torch.float16 else "float32",
            "model_id": HF_MODEL_ID,
            "chunk_seconds": CHUNK_SECONDS,
            "elapsed_seconds": round(t_total, 2),
            "elapsed_mmss": f"{int(t_total//60)}:{int(t_total%60):02d}",
            "pyannote_requested": ENABLE_DIARIZATION,
            "pyannote_loaded": bool(diar_pipeline),
            "pyannote_used": diar_used,
            "pyannote_load_error": diar_load_error,
            "pyannote_runtime_error": diar_error,
        }

        return JsonResponse(
            {
                "text": full_text,
                "segments": merged,          # รวม speaker + timestamps
                "diarization": diar_segments,  # ขอบเขตการพูดของแต่ละ speaker
                "debug": debug,
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    finally:
        for p in (tmp_in, tmp_wav):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
