import os
import requests
from dotenv import load_dotenv
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

# โหลดค่าใน .env
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "biodatlab/whisper-th-large-v2"   # หรือเปลี่ยนเป็น thonburian/pathumma



def index(request):
    return render(request, "index.html")   # ← เปลี่ยนจาก "speechtotext/index.html"

    # หรือถ้าไฟล์อยู่ใน speechtotext/templates/index.html
    # return render(request, "index.html")

@csrf_exempt
def transcribe(request):
    if request.method == "POST" and request.FILES.get("audio"):
        audio_file = request.FILES["audio"]

        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            data=audio_file.read()
        )

        try:
            return JsonResponse(response.json(), safe=False)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "No audio uploaded"}, status=400)
