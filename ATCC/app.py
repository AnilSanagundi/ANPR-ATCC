# app.py
import streamlit as st
import pandas as pd
import glob
import os
import altair as alt

st.set_page_config(page_title="ATCC Dashboard", layout="wide")

SUMMARY_FOLDER = "outputs/summaries"
CSV_FOLDER = "outputs/csvs"
VIDEO_OUTPUT_FOLDER = "outputs/videos"

st.title("🚦 ATCC Dashboard (YOLOv8)")

# load master summary if present
master_summary_path = os.path.join(SUMMARY_FOLDER, "master_summary.csv")
summary_files = sorted(glob.glob(os.path.join(SUMMARY_FOLDER, "*_summary.csv")))
video_csvs = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))

st.sidebar.header("Controls")

video_list = [os.path.basename(p) for p in video_csvs]
video_choice = st.sidebar.selectbox("Select video (CSV)", video_list if video_list else ["None"])

# select class for scatter
all_df = None
if video_choice and video_choice != "None":
    df = pd.read_csv(os.path.join(CSV_FOLDER, video_choice))
    st.header(f"Video: {video_choice}")
    st.write("Basic info:")
    st.write(f"Total tracked rows: {len(df)}")
    classes = sorted(df["class"].unique())
    class_choice = st.sidebar.selectbox("Select class for scatter plot", classes)
else:
    df = pd.DataFrame()
    classes = []
    class_choice = None

# TOP ROW: summary cards
col1, col2, col3, col4 = st.columns(4)
if os.path.exists(master_summary_path):
    master = pd.read_csv(master_summary_path)
    total_videos = len(master)
    total_vehicles = int(master["total_vehicles"].sum())
    total_frames = int(master["total_frames"].sum())
else:
    total_videos = len(video_csvs)
    total_vehicles = df["track_id"].nunique() if not df.empty else 0
    total_frames = df["frame"].max() if not df.empty else 0

col1.metric("Videos processed", total_videos)
col2.metric("Total tracked vehicles", total_vehicles)
col3.metric("Total frames processed", total_frames)
col4.metric("Unique classes", len(classes))

# ==== BAR PLOT ====
st.subheader("Class-wise Counts (Bar Plot)")
if os.path.exists(master_summary_path):
    master = pd.read_csv(master_summary_path)
    count_cols = [c for c in master.columns if c.startswith("count_")]

    if count_cols:
        bars = []
        for col in count_cols:
            cls = col.replace("count_", "")
            val = master[col].sum()
            bars.append({"class": cls, "count": int(val)})

        bar_df = pd.DataFrame(bars).sort_values("count", ascending=False)
        bar_chart = alt.Chart(bar_df).mark_bar().encode(
            x=alt.X('class:N', sort='-y'),
            y='count:Q',
            tooltip=['class', 'count']
        ).properties(width=800, height=350)
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.info("No count_* columns in master_summary.csv. Run processing script first.")
else:
    if not df.empty:
        agg = df.groupby("class")["track_id"].nunique().reset_index().rename(columns={"track_id":"count"})
        if not agg.empty:
            bar_chart = alt.Chart(agg).mark_bar().encode(
                x=alt.X('class:N', sort='-y'),
                y='count:Q',
                tooltip=['class', 'count']
            ).properties(width=800, height=350)
            st.altair_chart(bar_chart, use_container_width=True)
        else:
            st.write("No data yet. Run process_videos.py to generate CSVs.")
    else:
        st.write("No data found. Run process_videos.py to generate CSVs.")

# ==== LINE CHART ====
st.subheader("Vehicle Count Over Time (Line Chart)")
if not df.empty:
    per_frame = df.groupby("frame")["track_id"].nunique().reset_index().rename(columns={"track_id":"unique_tracks"})
    per_frame["time_s"] = per_frame["frame"] / (df["frame"].max() / (per_frame["frame"].max() or 1))
    line = alt.Chart(per_frame).mark_line(point=True).encode(
        x='frame:Q',
        y='unique_tracks:Q',
        tooltip=['frame','unique_tracks']
    ).properties(width=900, height=300)
    st.altair_chart(line, use_container_width=True)
else:
    st.info("No per-frame data. Run process_videos.py and refresh.")

# ==== SCATTER PLOT ====
st.subheader("Centroid Scatter Plot (Scatter)")
if not df.empty and class_choice:
    sub = df[df["class"] == class_choice]
    x_field = "time_s" if "time_s" in sub.columns else "frame"
    scatter = alt.Chart(sub).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X(f'{x_field}:Q', title='Time (s)' if x_field=="time_s" else "Frame"),
        y=alt.Y('cx:Q', title='Centroid X (px)'),
        color='track_id:N',
        tooltip=['frame','track_id','cx','cy','class']
    ).properties(width=900, height=400)
    st.altair_chart(scatter, use_container_width=True)
else:
    st.info("Select a video and class to see centroid scatter (left sidebar).")

# ==== TABLE ====
st.subheader("Per-frame Tracks Table (preview)")
if not df.empty:
    st.dataframe(df.head(200))
else:
    st.write("No data to show. Run process_videos.py first.")

# ==== VIDEO PREVIEW ADDED HERE ====
st.subheader("🎥 Detected Video Preview")

if video_choice and video_choice != "None":
    video_base = os.path.splitext(video_choice)[0]
    video_path = os.path.join(VIDEO_OUTPUT_FOLDER, f"{video_base}_out.mp4")

    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.warning("Processed video not found. Run process_videos.py to generate output video.")
else:
    st.info("Select a video to preview detections.")
