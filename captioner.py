"""Model, metrics and captioning logic.

Python imports this file only once, so the metrics are created once even
though Streamlit runs app.py again on every click.
"""
import io
import os
import sys
import threading
import time
import zipfile

import psutil
from PIL import Image
from prometheus_client import (Counter, Gauge, Histogram, Info, Summary,
                               disable_created_metrics, start_http_server)

MODEL_NAME = "Salesforce/blip-image-captioning-base"
METRICS_PORT = 18000
MAX_SIDE = 512
MAX_ZIP_IMAGES = 50
MAX_FILE_BYTES = 20 * 1024 * 1024
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

disable_created_metrics()

REQUESTS = Counter("caption_requests", "Caption requests, one per image or ZIP",
                   ["mode", "client_ip", "session_id", "browser"])
IMAGES = Counter("caption_images_processed", "Images processed",
                 ["mode", "status", "client_ip"])
ERRORS = Counter("caption_errors", "Errors while processing uploads",
                 ["mode", "error_type"])
SKIPPED = Counter("caption_files_skipped", "Files in a ZIP that were not captioned",
                  ["reason"])

ACTIVE = Gauge("caption_active_requests", "Requests being processed now", ["mode"])
MODEL_LOADED = Gauge("caption_model_loaded", "1 when the model is loaded")
MODEL_LOAD_SECONDS = Gauge("caption_model_load_seconds", "Time taken to load the model")
MODEL_MEMORY = Gauge("caption_model_memory_bytes", "Memory added by loading the model")
PROCESS_MEMORY = Gauge("caption_process_memory_bytes", "Memory (RSS) of the app process")
PROCESS_MEMORY.set_function(lambda: psutil.Process().memory_info().rss)
BULK_REMAINING = Gauge("caption_bulk_images_remaining", "Images waiting in running ZIP jobs")
LAST_INFERENCE = Gauge("caption_last_inference_seconds", "Inference time of the latest image",
                       ["mode"])
LAST_BATCH_SIZE = Gauge("caption_last_batch_size", "Images in the latest ZIP")

INFERENCE_SECONDS = Histogram("caption_inference_seconds", "Inference time per image",
                              ["mode"],
                              buckets=(0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 30))
BULK_JOB_SECONDS = Histogram("caption_bulk_job_seconds", "Time to caption one ZIP",
                             buckets=(5, 10, 20, 30, 60, 90, 120, 180, 300, 600))
IMAGE_MEGAPIXELS = Histogram("caption_image_megapixels", "Size of uploaded images",
                             ["mode"], buckets=(0.1, 0.3, 0.5, 1, 2, 4, 8, 12, 20))
CAPTION_WORDS = Histogram("caption_length_words", "Words in generated captions",
                          ["mode"], buckets=(3, 5, 7, 9, 11, 13, 15, 20))

UPLOAD_BYTES = Summary("caption_upload_bytes", "Size of uploaded files", ["mode"])
PREPROCESS_SECONDS = Summary("caption_preprocess_seconds", "Time to resize an image",
                             ["mode"])
ZIP_IMAGES = Summary("caption_zip_images_per_job", "Images found per ZIP")

APP_INFO = Info("caption_app", "Image captioning app build information")
APP_INFO.info({"model": MODEL_NAME, "device": "cpu", "version": "1.0",
               "python": sys.version.split()[0]})

# create the label combinations at 0 so Prometheus sees the first increase too
for mode in ("single", "bulk"):
    ACTIVE.labels(mode)
    INFERENCE_SECONDS.labels(mode)
    PREPROCESS_SECONDS.labels(mode)
    UPLOAD_BYTES.labels(mode)
    IMAGE_MEGAPIXELS.labels(mode)
    CAPTION_WORDS.labels(mode)
for mode, kinds in (("single", ("bad_image", "inference_error")),
                    ("bulk", ("bad_image", "bad_zip", "no_images", "inference_error"))):
    for kind in kinds:
        ERRORS.labels(mode, kind)
for reason in ("hidden_file", "not_image", "too_large", "over_limit"):
    SKIPPED.labels(reason)

_server_lock = threading.Lock()
_server_started = False
_model_lock = threading.Lock()
_model = None


def register_session(client):
    for mode in ("single", "bulk"):
        REQUESTS.labels(mode, client["ip"], client["session"], client["browser"])
        for status in ("success", "failed"):
            IMAGES.labels(mode, status, client["ip"])


def start_metrics_server():
    global _server_started
    with _server_lock:
        if not _server_started:
            start_http_server(METRICS_PORT)
            _server_started = True


def load_model():
    global _model
    with _model_lock:
        if _model is None:
            import torch
            from transformers import AutoProcessor, BlipForConditionalGeneration

            # keep one core free so the VM and the monitoring tools stay responsive
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            rss_before = psutil.Process().memory_info().rss
            start = time.perf_counter()
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
            model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
            model.eval()
            MODEL_LOAD_SECONDS.set(time.perf_counter() - start)
            MODEL_MEMORY.set(max(0, psutil.Process().memory_info().rss - rss_before))
            MODEL_LOADED.set(1)
            _model = (processor, model)
    return _model


def caption_image(img, mode):
    import torch

    processor, model = load_model()
    start = time.perf_counter()
    img = img.convert("RGB")
    IMAGE_MEGAPIXELS.labels(mode).observe(img.width * img.height / 1e6)
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    inputs = processor(images=img, return_tensors="pt")
    PREPROCESS_SECONDS.labels(mode).observe(time.perf_counter() - start)

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=30)
    seconds = time.perf_counter() - start
    caption = processor.decode(output[0], skip_special_tokens=True).strip()

    INFERENCE_SECONDS.labels(mode).observe(seconds)
    LAST_INFERENCE.labels(mode).set(seconds)
    CAPTION_WORDS.labels(mode).observe(len(caption.split()))
    return caption, seconds


def process_single(data, client):
    REQUESTS.labels("single", client["ip"], client["session"], client["browser"]).inc()
    UPLOAD_BYTES.labels("single").observe(len(data))
    with ACTIVE.labels("single").track_inprogress():
        try:
            img = Image.open(io.BytesIO(data))
            caption, seconds = caption_image(img, "single")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            IMAGES.labels("single", "failed", client["ip"]).inc()
            ERRORS.labels("single", "bad_image").inc()
            return {"ok": False, "error": f"Could not read this image: {exc}"}
        except Exception as exc:
            IMAGES.labels("single", "failed", client["ip"]).inc()
            ERRORS.labels("single", "inference_error").inc()
            return {"ok": False, "error": f"Captioning failed: {exc}"}
        IMAGES.labels("single", "success", client["ip"]).inc()
        return {"ok": True, "caption": caption, "seconds": seconds}


def list_zip_images(zf):
    images = []
    for info in zf.infolist():
        name = info.filename
        base = os.path.basename(name)
        if info.is_dir():
            continue
        if name.startswith("__MACOSX/") or base.startswith("."):
            SKIPPED.labels("hidden_file").inc()
        elif not base.lower().endswith(IMAGE_EXTS):
            SKIPPED.labels("not_image").inc()
        elif info.file_size > MAX_FILE_BYTES:
            SKIPPED.labels("too_large").inc()
        elif len(images) >= MAX_ZIP_IMAGES:
            SKIPPED.labels("over_limit").inc()
        else:
            images.append(name)
    return images


def process_zip(data, client, on_progress=None):
    REQUESTS.labels("bulk", client["ip"], client["session"], client["browser"]).inc()
    UPLOAD_BYTES.labels("bulk").observe(len(data))
    job_start = time.perf_counter()
    with ACTIVE.labels("bulk").track_inprogress():
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            ERRORS.labels("bulk", "bad_zip").inc()
            return {"ok": False, "error": "This file is not a valid ZIP.", "results": []}

        names = list_zip_images(zf)
        ZIP_IMAGES.observe(len(names))
        LAST_BATCH_SIZE.set(len(names))
        if not names:
            ERRORS.labels("bulk", "no_images").inc()
            return {"ok": False, "error": "No images found in this ZIP.", "results": []}

        results = []
        left = len(names)
        BULK_REMAINING.inc(left)
        try:
            for i, name in enumerate(names, start=1):
                try:
                    with zf.open(name) as fh:
                        img = Image.open(io.BytesIO(fh.read()))
                        caption, seconds = caption_image(img, "bulk")
                    IMAGES.labels("bulk", "success", client["ip"]).inc()
                    results.append({"file": name, "caption": caption,
                                    "seconds": round(seconds, 2)})
                except (OSError, ValueError, Image.DecompressionBombError,
                        zipfile.BadZipFile):
                    IMAGES.labels("bulk", "failed", client["ip"]).inc()
                    ERRORS.labels("bulk", "bad_image").inc()
                    results.append({"file": name, "caption": "ERROR: unreadable image",
                                    "seconds": None})
                except Exception as exc:
                    IMAGES.labels("bulk", "failed", client["ip"]).inc()
                    ERRORS.labels("bulk", "inference_error").inc()
                    results.append({"file": name, "caption": f"ERROR: {exc}",
                                    "seconds": None})
                left -= 1
                BULK_REMAINING.dec()
                if on_progress:
                    on_progress(i, len(names), name)
        finally:
            # if the job stops early, remove the images it never reached
            BULK_REMAINING.dec(left)

        BULK_JOB_SECONDS.observe(time.perf_counter() - job_start)
        return {"ok": True, "results": results}
