"""CPU image captioning app. Run with: streamlit run app.py"""
import csv
import io
import uuid

import streamlit as st

import captioner

st.set_page_config(page_title="Image Captioning", page_icon="🖼️")
captioner.start_metrics_server()


@st.cache_resource(show_spinner="Loading the captioning model, the first run downloads about 1 GB...")
def get_model():
    return captioner.load_model()


def client_info():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]
    # Streamlit gives None on localhost
    ip = st.context.ip_address or "127.0.0.1"
    agent = st.context.headers.get("User-Agent", "")
    if "Edg/" in agent:
        browser = "edge"
    elif "Firefox" in agent:
        browser = "firefox"
    elif "Chrome" in agent:
        browser = "chrome"
    elif "Safari" in agent:
        browser = "safari"
    else:
        browser = "other"
    return {"ip": ip, "session": st.session_state.session_id, "browser": browser}


get_model()
client = client_info()
captioner.register_session(client)

st.title("CPU Image Captioning")
st.caption(f"Model: {captioner.MODEL_NAME} | metrics on port {captioner.METRICS_PORT}")

single_tab, bulk_tab = st.tabs(["Single image", "Bulk (ZIP)"])

with single_tab:
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                              key="single_upload")
    if upload is not None:
        st.image(upload, width="stretch")
        if st.button("Generate caption", key="single_button"):
            with st.spinner("Captioning..."):
                result = captioner.process_single(upload.getvalue(), client)
            if result["ok"]:
                st.success(result["caption"])
                st.caption(f"Inference time: {result['seconds']:.2f} s")
            else:
                st.error(result["error"])

with bulk_tab:
    st.write(f"Upload a ZIP of images. Up to {captioner.MAX_ZIP_IMAGES} images are "
             "captioned per ZIP and other files are skipped.")
    zip_upload = st.file_uploader("Upload a ZIP file", type=["zip"], key="bulk_upload")
    if zip_upload is not None and st.button("Caption all images", key="bulk_button"):
        bar = st.progress(0.0, text="Starting...")

        def show_progress(done, total, name):
            bar.progress(done / total, text=f"{done}/{total} done: {name}")

        result = captioner.process_zip(zip_upload.getvalue(), client, show_progress)
        if not result["ok"]:
            st.error(result["error"])
        else:
            st.success(f"Captioned {len(result['results'])} images.")
            st.dataframe(result["results"], width="stretch")
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=["file", "caption", "seconds"])
            writer.writeheader()
            writer.writerows(result["results"])
            st.download_button("Download captions (CSV)", out.getvalue(),
                               file_name="captions.csv", mime="text/csv")
