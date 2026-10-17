# Vision server image for the Vultr GPU instance.
# Build from the REPO ROOT (hand-detection/ is copied in for the glove band reader):
#   docker build -f object-detection/Dockerfile -t glance-vision .
# Run:
#   docker run -d --gpus all --restart unless-stopped -p 8000:8000 --env-file object-detection/.env --name glance-vision glance-vision
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "transformers>=4.56" fastapi "uvicorn[standard]" pillow pyyaml numpy websockets \
    opencv-contrib-python-headless

# Bake the weights into the image so the instance never downloads at startup.
# Keep these ids in sync with object-detection/config.yaml.
ARG DETECTOR_ID=IDEA-Research/grounding-dino-tiny
ARG TRACKER_ID=facebook/sam2.1-hiera-tiny
RUN python -c "from huggingface_hub import snapshot_download as d; d('${DETECTOR_ID}'); d('${TRACKER_ID}')"

WORKDIR /app
COPY hand-detection /app/hand-detection
COPY object-detection /app/object-detection
WORKDIR /app/object-detection

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "5"]
