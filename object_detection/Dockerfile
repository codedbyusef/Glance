# Vision server image for the Vultr GPU instance.
# Build from the REPO ROOT (hand_detection/ and depth_estimation/ are copied in next to object_detection/):
#   docker build -f object_detection/Dockerfile -t glance-vision .
# Run:
#   docker run -d --gpus all --restart unless-stopped -p 8000:8000 --env-file object_detection/.env --name glance-vision glance-vision
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
# Keep these ids in sync with object_detection/config.yaml.
ARG DETECTOR_ID=IDEA-Research/grounding-dino-tiny
ARG TRACKER_ID=facebook/sam2.1-hiera-tiny
RUN python -c "from huggingface_hub import snapshot_download as d; d('${DETECTOR_ID}'); d('${TRACKER_ID}')"

# The packages import each other package-qualified (object_detection, hand_detection, depth_estimation),
# so they sit side by side under /app, which is the working directory and on sys.path.
WORKDIR /app
COPY hand_detection /app/hand_detection
COPY depth_estimation /app/depth_estimation
COPY object_detection /app/object_detection

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "object_detection.server:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "5"]
