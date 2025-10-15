import os, tempfile, librosa, torch, soundfile as sf
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from transformers import pipeline

# โหลดโมเดล (ใช้ Pathumma / Whisper Thai)
MODEL_ID = "biodatlab/whisper-th-small"  # เปลี่ยนเป็น whisper-th-large-v3-combined ได้
device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_ID,
    chunk_length_s=15,
    device=device
)

def index(request):
    return render(request, "speechtotext/index.html")

@csrf_exempt
@require_POST
def speech_to_text(request):
    try:
        f = request.FILES.get("file")
        if not f:
            return JsonResponse({"error": "Missing file (field name: file)"}, status=400)

        name = getattr(f, "name", "") or ""
        ext = os.path.splitext(name)[1].lower() or ".mp3"

        tmp_in, tmp_wav = None, None
        try:
            # เขียนไฟล์ที่อัปโหลดมาเป็นไฟล์ชั่วคราว
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as t:
                tmp_in = t.name
                for chunk in f.chunks():
                    t.write(chunk)

            # แปลงเสียงเป็น mono 16k
            y, sr = librosa.load(tmp_in, sr=16000, mono=True)
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(tmp_wav, y, 16000, subtype="PCM_16")

            # รันโมเดล
            result = asr(tmp_wav, return_timestamps=False)
        finally:
            for p in [tmp_in, tmp_wav]:
                if p and os.path.exists(p):
                    try: os.remove(p)
                    except OSError: pass

        return JsonResponse({
            "text": result.get("text", ""),
            "segments": result.get("chunks") or result.get("segments")
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
