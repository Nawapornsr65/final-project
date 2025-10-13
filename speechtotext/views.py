# speechtotext/views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import io, librosa
from transformers import pipeline
import torch

# ===== โหลดโมเดลครั้งเดียว =====
device = 0 if torch.cuda.is_available() else -1
asr = pipeline(
    task="automatic-speech-recognition",
    model="biodatlab/whisper-th-large-v3-combined",
    chunk_length_s=30,
    device=device
)

def index(request):
    # ต้องมีไฟล์ templates/speechtotext/index.html
    return render(request, "speechtotext/index.html")

@csrf_exempt
@require_POST
def speech_to_text(request):
    try:
        f = request.FILES.get("file")
        if not f:
            return JsonResponse({"error": "Missing file (field name: file)"}, status=400)

        raw = f.read()
        y, sr = librosa.load(io.BytesIO(raw), sr=16000, mono=True)
        result = asr({"array": y, "sampling_rate": 16000}, return_timestamps=True)

        return JsonResponse({
            "text": result["text"],
            "segments": result.get("chunks") or result.get("segments")
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
