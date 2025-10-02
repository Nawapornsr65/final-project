import os
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv

load_dotenv()  # โหลดค่าในไฟล์ .env
HF_API_KEY = os.getenv("HF_API_KEY")  # ดึง Token

@csrf_exempt
def transcribe(request):
    if request.method == "POST":
        audio_file = request.FILES.get("audio")
        if not audio_file:
            return JsonResponse({"error": "No audio uploaded"}, status=400)

        response = requests.post(
            "https://api-inference.huggingface.co/models/Thonburian/whisper-large-v3-th",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            data=audio_file.read()
        )

        return JsonResponse(response.json())
    return JsonResponse({"error": "Invalid request"}, status=400)
