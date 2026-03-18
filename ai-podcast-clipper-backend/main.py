import json
import pathlib
import shutil
import subprocess
import time
import uuid
import os

import modal
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel


class ProcessVideoRequest(BaseModel):
    s3_key: str


# Create container image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "git"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands([
        "mkdir -p /usr/share/fonts/truetype/custom",
        "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
        "fc-cache -f -v"
    ])
    .add_local_dir("asd", "/asd", copy=True)
)


# Modal App
app = modal.App("ai-podcast-clipper", image=image)


# Model cache volume
volume = modal.Volume.from_name(
    "ai-podcast-clipper-cache",
    create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()


@app.cls(
    gpu="L40S",
    timeout=900,
    retries=0,
    scaledown_window=20,
    secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")],
    volumes={mount_path: volume}
)
class AiPodcastClipper:

    @modal.enter()
    def load_model(self):
        import whisperx
        print("Loading models...")

        self.whisperx_model = whisperx.load_model(
            "large-v2",
            device="cuda",
            compute_type="float16",
            language="en"
        )

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en",
            device="cuda"
        )

        print("Transcription models loaded...")

    def transcribe_video(self, base_dir: pathlib.Path, video_path: pathlib.Path) -> str:

        audio_path = base_dir / "audio.wav"

        extract_cmd = ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)]
        subprocess.run(extract_cmd, check=True, capture_output=True)

        import whisperx
        print("Starting transcription with WhisperX...")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))

        result = self.whisperx_model.transcribe(audio, batch_size=16)

        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

        duration = time.time() - start_time
        print(f"Transcription and alignment took {duration:.2f} seconds")

        print("Segments detected:", len(result["segments"]))

        return json.dumps(result)

    @modal.fastapi_endpoint(method="POST")
    def process_video(
        self,
        request: ProcessVideoRequest,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme)
    ):

        s3_key = request.s3_key

        # Verify bearer token
        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Create unique run directory
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # Video download path
        video_path = base_dir / "input.mp4"

        import boto3

        # Create S3 client with credentials
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="ap-south-1"
        )

        # Download video
        try:
            s3_client.download_file(
                "ai-podcast-clipper",
                s3_key,
                str(video_path)
            )

            print("Downloaded files:", os.listdir(base_dir))

        # Transcribe video
            transcript = self.transcribe_video(base_dir, video_path)

            return {
                "status": "success",
                "downloaded_file": str(video_path),
                "transcript": json.loads(transcript)
            }
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)


@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper = AiPodcastClipper()

    # Updated method to avoid deprecation
    url = ai_podcast_clipper.process_video.get_web_url()

    payload = {
        "s3_key": "test1/mi65min.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers
    )

    response.raise_for_status()

    result = response.json()
    print(result)