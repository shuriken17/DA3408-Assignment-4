# DA3408 AI Operations — Assignment 4

**Prabhav Gupta DA24B018**

A CPU only image captioning app with Streamlit and BLIP, monitored with Prometheus, node_exporter, Alertmanager and Grafana.

## Files

- `app.py` — the Streamlit app, with a Single image tab and a Bulk ZIP tab
- `captioner.py` — the model, the metrics and the captioning
- `requirements.txt` — Python packages
- `prometheus.yml` — scrapes the app, node_exporter and Prometheus
- `rules.yml` — recording rules and alerts
- `alertmanager.yml` — sends alerts to a webhook
- `grafana_dashboard.json` — the exported dashboard

## Setup

```bash
conda create -n a4 python=3.12 -y
conda activate a4
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The BLIP model, about 1 GB, downloads on the first run.

Get Prometheus, Alertmanager and Node Exporter from https://prometheus.io/download/ and extract them into `~/monitoring_tools`. I used Prometheus 3.15.0, Alertmanager 0.34.1 and Node Exporter 1.12.1.

Start Grafana once with the command below, and use `docker start grafana` after that. Log in at http://localhost:3000 with admin and admin, and add a Prometheus data source at http://172.17.0.1:9090.

```bash
docker run -d -p 3000:3000 --name=grafana grafana/grafana-enterprise
```

Put your own webhook address from https://webhook.site in `alertmanager.yml`.

## Run

Each command needs its own terminal. They assume the repo is in `~/Assignment_4`.

app:

```bash
cd ~/Assignment_4 && conda activate a4 && streamlit run app.py
```

node_exporter:

```bash
cd ~/monitoring_tools/node_exporter-1.12.1.linux-amd64 && ./node_exporter
```

alertmanager:

```bash
cd ~/monitoring_tools/alertmanager-0.34.1.linux-amd64 && ./alertmanager --config.file=$HOME/Assignment_4/alertmanager.yml
```

prometheus:

```bash
cd ~/monitoring_tools/prometheus-3.15.0.linux-amd64 && ./prometheus --config.file=$HOME/Assignment_4/prometheus.yml
```

The app is on http://localhost:8501 and its metrics are on http://localhost:18000/metrics. Open the app page once, because the metrics port only opens when a page runs the app. Then import `grafana_dashboard.json` in Grafana from Dashboards, New, Import. Any ZIP of JPG or PNG images works for the Bulk tab.

## Metrics

| Metric | Type | Labels | Measures |
|---|---|---|---|
| `caption_requests_total` | Counter | mode, client_ip, session_id, browser | Uploads |
| `caption_images_processed_total` | Counter | mode, status, client_ip | Images done |
| `caption_errors_total` | Counter | mode, error_type | Errors |
| `caption_files_skipped_total` | Counter | reason | Skipped ZIP files |
| `caption_active_requests` | Gauge | mode | Requests running now |
| `caption_bulk_images_remaining` | Gauge | none | Images waiting in ZIPs |
| `caption_process_memory_bytes` | Gauge | none | App memory |
| `caption_model_memory_bytes` | Gauge | none | Memory added by the model |
| `caption_model_load_seconds` | Gauge | none | Model load time |
| `caption_model_loaded` | Gauge | none | 1 once loaded |
| `caption_last_batch_size` | Gauge | none | Images in the last ZIP |
| `caption_last_inference_seconds` | Gauge | mode | Time of the last image |
| `caption_inference_seconds` | Histogram | mode | Inference time per image |
| `caption_bulk_job_seconds` | Histogram | none | Time per ZIP |
| `caption_image_megapixels` | Histogram | mode | Image size |
| `caption_length_words` | Histogram | mode | Words per caption |
| `caption_upload_bytes` | Summary | mode | Upload size |
| `caption_preprocess_seconds` | Summary | mode | Resize time |
| `caption_zip_images_per_job` | Summary | none | Images per ZIP |
| `caption_app_info` | Info | model, device, version, python | Model and build details |

The mode label is single or bulk.

## Alerts and dashboard

Alerts: CaptionAppDown, NodeExporterDown, HighCPUUtilisation, HighMemoryUtilisation, SlowInference and CaptionErrors. When I stopped the app, CaptionAppDown fired after 30 seconds and reached the webhook. It resolved after I restarted the app.

The dashboard goes from Health to Throughput, Latency, App internals, System and Correlation, and refreshes every 5 seconds. In a load test with two 50 image ZIPs, machine CPU peaked at 78.2 percent, the app at 59.5 percent and bulk throughput at 40.4 images per minute.

## AI usage

I used Claude in this project. My prompts asked it to write the code, config files, dashboard and PromQL, help me debug and assist me with problems. I ran everything on my machine and tested it myself.

The comments in `rules.yml` and the table below say what each query was asked to give. The other dashboard queries are simple variants of these.

| Panel | PromQL | Asked to give |
|---|---|---|
| Failed images | `(sum(increase(caption_images_processed_total{status="failed"}[5m])) or vector(0)) / clamp_min(sum(increase(caption_images_processed_total[5m])) or vector(0), 1)` | Share of images that failed in 5 minutes |
| Model status | `(max(caption_model_loaded) * scalar(up{job="caption_app"})) or vector(0)` | 1 when the app is up and the model is loaded |
| App uptime | `max((time() - process_start_time_seconds{job="caption_app"}) and on() (up{job="caption_app"} == 1)) or vector(0)` | Seconds since the app started, 0 when down |
| Images per minute | `sum by (mode, status) (rate(caption_images_processed_total[1m])) * 60` | Images per minute by mode and result |
| Sessions | `sum by (session_id, client_ip, browser, mode) (caption_requests_total) > 0` | Requests per browser session |
| Inference latency | `histogram_quantile(0.95, sum by (le, mode) (rate(caption_inference_seconds_bucket[1m])))` | 95th percentile inference time. The median uses 0.5 |
| Average time per image | `sum by (mode) (rate(caption_inference_seconds_sum[1m])) / sum by (mode) (rate(caption_inference_seconds_count[1m]))` | Average inference time per image |
| Bulk job duration | `histogram_quantile(0.95, sum by (le) (rate(caption_bulk_job_seconds_bucket[5m])))` | 95th percentile time to finish one ZIP |
| CPU by mode | `100 * sum(rate(node_cpu_seconds_total{mode=~"user\|nice"}[1m])) / scalar(count(count by (cpu) (node_cpu_seconds_total)))` | CPU percent of all cores in user mode. System and waiting use other modes |
| CPU unassigned | `(sum(increase(caption_images_processed_total{status="failed"}[5m])) or vector(0)) / clamp_min(sum(increase(caption_images_processed_total[5m])) or vector(0), 1)` | Machine CPU minus the counted modes |
| Machine CPU | `100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m])))` | Total machine CPU percent |
| App CPU | `100 * rate(process_cpu_seconds_total{job="caption_app"}[1m]) / scalar(count(count by (cpu) (node_cpu_seconds_total)))` | CPU percent of all cores used by the app |
| Disk I/O | `sum(rate(node_disk_read_bytes_total{device!~"loop.*\|ram.*"}[1m]))` | Disk read bytes per second. Writes use the written counter |

This repo is private until the deadline and will be made public after.
