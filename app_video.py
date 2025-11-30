import streamlit as st
import os
import tempfile
import time
import pandas as pd

# Import the processing function from your script
from video_frame_by_frame_tracker import process_video

st.set_page_config(page_title="Waste Video Processor", layout="centered")
st.title("♻️ Waste Video Processor — Frame-by-frame demo")

st.markdown("""
Upload a short video (<= 15 seconds). The app will:
- run frame-by-frame detection using the trained model,
- draw bounding boxes for moving objects and predicted labels,
- provide a summary CSV with counts and average confidence.
""")

uploaded = st.file_uploader("Upload MP4/AVI video (<= 15 s)", type=["mp4","avi","mov"])
frame_skip = st.slider("Frame skip (1 = all frames, larger = faster)", 1, 6, 2)

if uploaded:
    tdir = tempfile.mkdtemp()
    in_path = os.path.join(tdir, uploaded.name)
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # quick ffprobe-like check to get duration (approx using cv2)
    import cv2
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / (fps if fps>0 else 30.0)
    cap.release()

    if duration > 15.0:
        st.error(f"Video is {duration:.1f}s long — limit is 15s. Please upload a shorter clip.")
    else:
        st.info(f"Video length: {duration:.1f}s  — frames: {total_frames} — fps: {fps:.1f}")

        out_video = os.path.join(tdir, "output_processed.mp4")
        out_csv = os.path.join(tdir, "predictions.csv")
        summary_csv = os.path.join(tdir, "summary.csv")

        if st.button("Start processing"):
            with st.spinner("Processing video (this can take a while)..."):
                t0 = time.time()
                res = process_video(
                    video_in=in_path,
                    video_out=out_video,
                    out_csv=out_csv,
                    summary_csv=summary_csv,
                    model_path="models/EfficientNetB2_savedmodel",
                    frame_skip=frame_skip,
                    max_duration_sec=15
                )
                elapsed = time.time() - t0
            st.success(f"Done in {elapsed:.1f}s — processed frames: {res.get('processed_frames')}")
            if os.path.exists(out_video):
                st.video(out_video)
                with open(out_video, "rb") as f:
                    st.download_button("Download processed video", f, file_name="processed_video.mp4")
            if os.path.exists(summary_csv):
                st.write("Summary counts:")
                st.dataframe(pd.read_csv(summary_csv))
                with open(summary_csv, "rb") as f:
                    st.download_button("Download summary CSV", f, file_name="summary.csv")
            if os.path.exists(out_csv):
                st.write("Per-frame predictions (first rows):")
                st.dataframe(pd.read_csv(out_csv).head(200))
