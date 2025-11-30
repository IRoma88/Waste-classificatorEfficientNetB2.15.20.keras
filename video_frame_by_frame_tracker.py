import os
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import tensorflow as tf

# --- EDITA SI CAMBIAS ETIQUETAS ---
DISPLAY_LABELS = [
    "Blue - Cardboard & Briks",
    "Blue - Glass Bottles (Type 1)",
    "Blue - Glass Bottles (Type 2)",
    "Blue - Metal Cans & Tupperware",
    "Blue - Paper & Books",
    "Blue - Plastics (Type 1)",
    "Blue - Plastics (Type 2)",
    "Brown - Organic",
    "Gray - General Trash",
    "Take-Back Shop",
    "HHW",
    "Medical Waste",
    "Drop-off Items"
]

# helper: load serving signature robustly
def _load_model(model_path):
    model = tf.saved_model.load(model_path)
    sigs = list(model.signatures.keys())
    # prefer serving_default, else first
    signature_name = "serving_default" if "serving_default" in sigs else (sigs[0] if len(sigs)>0 else None)
    if signature_name is None:
        raise RuntimeError("No signature found in SavedModel.")
    infer = model.signatures[signature_name]
    return model, infer

def process_video(
    video_in,
    video_out,
    out_csv=None,
    summary_csv=None,
    model_path="models/EfficientNetB2_savedmodel",
    img_size=(380,380),
    frame_skip=2,
    max_duration_sec=15,
    conf_threshold=0.05,
    min_contour_area=800
):
    """
    Process video frame-by-frame:
     - limit to max_duration_sec
     - process every frame_skip frame
     - detect motion regions (background subtractor)
     - run model inference on full frame (you can change to crop per box in future)
     - draw boxes for moving regions when model has reasonable confidence
    Returns dict with paths and stats.
    """
    t0_all = time.time()

    if not os.path.exists(video_in):
        raise FileNotFoundError(video_in)

    # load model
    print("Loading SavedModel...", model_path)
    model, infer = _load_model(model_path)
    print("Signatures:", list(model.signatures.keys()))
    print("Using signature:", infer)

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_in)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(min(total_frames, fps * max_duration_sec))

    # writer (same size but we optionally add a small right panel)
    panel_w = 360
    outW = W + panel_w
    outH = H
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(video_out, fourcc, fps, (outW, outH))

    backSub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=48, detectShadows=False)

    records = []
    counts = {}
    sum_conf = {}

    frame_idx = 0
    processed = 0
    t0 = time.time()

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # optional skip for speed
        if (frame_idx % frame_skip) != 0:
            frame_idx += 1
            continue

        # motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        fg = backSub.apply(blur)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.dilate(fg, None, iterations=2)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < min_contour_area:
                continue
            x,y,w,h = cv2.boundingRect(c)
            boxes.append((x,y,w,h))

        # prepare frame for inference: resize full frame (we use full-frame inference for simplicity)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil = pil.resize(img_size)
        arr = np.expand_dims(np.array(pil).astype("float32")/255.0, axis=0)
        # some SavedModel expect preprocessing; if you used Keras preprocess_input, adapt here.

        # run inference
        preds_dict = infer(tf.constant(arr))
        preds = list(preds_dict.values())[0].numpy()[0]
        pred_idx = int(np.argmax(preds))
        pred_conf = float(np.max(preds))
        pred_label = DISPLAY_LABELS[pred_idx] if pred_idx < len(DISPLAY_LABELS) else f"cls_{pred_idx}"

        timestamp = frame_idx / fps

        # record
        records.append({
            "frame_index": frame_idx,
            "timestamp_sec": float(timestamp),
            "pred_index": pred_idx,
            "pred_label": pred_label,
            "confidence": float(pred_conf)
        })

        # update counts for summary (we'll count per box below)
        if pred_label not in counts:
            counts[pred_label] = 0
            sum_conf[pred_label] = 0.0

        # draw boxes if motion + good confidence
        display_label = pred_label if pred_conf >= conf_threshold else "UNCERTAIN"
        display_conf = pred_conf if pred_conf >= conf_threshold else 0.0

        if boxes and display_label != "UNCERTAIN":
            # add counts per moving box (approximate)
            for (x,y,w,h) in boxes:
                counts[display_label] += 1
                sum_conf[display_label] += display_conf
                # draw rectangle
                cv2.rectangle(frame, (x,y), (x+w, y+h), (16, 200, 200), 2)
                txt = f"{display_label} {int(display_conf*100)}%"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x, y-th-8), (x+tw+6, y), (16,200,200), -1)
                cv2.putText(frame, txt, (x+3, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        else:
            # show full-frame label on top-left
            txt = f"{display_label} {int(display_conf*100)}%"
            cv2.rectangle(frame, (6,6), (260,46), (0,0,0), -1)
            cv2.putText(frame, txt, (12,34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # right panel: simple counts text
        panel = np.zeros((H, panel_w, 3), dtype=np.uint8) + 30
        cv2.putText(panel, "Summary (boxes)", (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 1, cv2.LINE_AA)
        y0 = 60
        line_h = 24
        sorted_keys = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
        for i, k in enumerate(sorted_keys[:12]):
            txt = f"{k[:24]:24s} {counts[k]:5d}  avg {int((sum_conf[k]/counts[k])*100) if counts[k]>0 else 0:3d}%"
            cv2.putText(panel, txt, (8, y0 + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)

        # add small progress bar
        prog_w = int((frame_idx / max(1, max_frames)) * (W-20))
        cv2.rectangle(frame, (10, H-32), (W-10, H-12), (50,50,50), -1)
        cv2.rectangle(frame, (10, H-32), (10+prog_w, H-12), (30,180,30), -1)

        out_frame = np.concatenate([frame, panel], axis=1)
        writer.write(out_frame)

        processed += 1
        frame_idx += frame_skip

        # print progress to console for Streamlit reading
        if processed % 20 == 0:
            elapsed = time.time() - t0
            print(f"Processed frame {frame_idx}/{max_frames} | processed {processed} frames | elapsed {elapsed:.1f}s")
        else:
            # minimal progress ping
            print(f"PF {frame_idx}/{max_frames}")

    # cleanup
    cap.release()
    writer.release()

    # save CSVs
    if out_csv:
        df = pd.DataFrame.from_records(records)
        df.to_csv(out_csv, index=False)
    if summary_csv:
        rows = []
        for k in counts:
            rows.append({
                "label": k,
                "boxes": counts[k],
                "avg_conf": (sum_conf[k]/counts[k]) if counts[k]>0 else 0.0
            })
        sdf = pd.DataFrame(rows).sort_values("boxes", ascending=False)
        sdf.to_csv(summary_csv, index=False)

    t_total = time.time() - t0_all
    print("Done. processed frames:", processed, "time(s):", round(t_total,1))
    return {"video_out": video_out, "out_csv": out_csv, "summary_csv": summary_csv, "processed_frames": processed}

# CLI
def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_in", required=True)
    parser.add_argument("--video_out", required=True)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--model_path", default="models/EfficientNetB2_savedmodel")
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--max_duration", type=int, default=15)
    args = parser.parse_args()

    process_video(
        args.video_in,
        args.video_out,
        out_csv=args.out_csv,
        summary_csv=args.summary_csv,
        model_path=args.model_path,
        frame_skip=args.frame_skip,
        max_duration_sec=args.max_duration
    )

if __name__ == "__main__":
    _cli()
