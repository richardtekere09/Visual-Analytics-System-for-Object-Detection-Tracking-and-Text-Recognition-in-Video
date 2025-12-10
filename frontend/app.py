import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import cv2
import numpy as np
from io import BytesIO

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Visual Analytics System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_video(video_file):
    """Upload video to API"""
    files = {"file": (video_file.name, video_file, video_file.type)}
    response = requests.post(f"{API_URL}/upload", files=files)
    return response.json()


def start_processing(video_id):
    """Start video processing"""
    response = requests.post(f"{API_URL}/process/{video_id}")
    return response.json()


def get_status(video_id):
    """Get processing status"""
    response = requests.get(f"{API_URL}/status/{video_id}")
    return response.json()


def get_results(video_id):
    """Get processing results"""
    response = requests.get(f"{API_URL}/results/{video_id}")
    return response.json()


def main():
    """Main application"""

    # Header
    st.markdown(
        '<h1 class="main-header">🎥 Visual Analytics System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("### Object Detection, Tracking, and Text Recognition in Video")

    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running! Please start the backend server.")
        st.code("cd backend && python -m app.main", language="bash")
        return

    st.success("✅ API is running")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page", ["Upload & Process", "View Results", "Analytics Dashboard"]
    )

    if page == "Upload & Process":
        upload_and_process_page()
    elif page == "View Results":
        view_results_page()
    else:
        analytics_dashboard_page()


def upload_and_process_page():
    """Upload and process video page"""
    st.header("📤 Upload & Process Video")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV",
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**Filename:** {uploaded_file.name}")
            st.info(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")

        if st.button("🚀 Upload and Process", type="primary", use_container_width=True):
            with st.spinner("Uploading video..."):
                # Upload
                try:
                    upload_response = upload_video(uploaded_file)
                    video_id = upload_response["video_id"]

                    st.success(f"✅ Video uploaded! ID: `{video_id}`")

                    # Display metadata
                    metadata = upload_response["metadata"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Duration", f"{metadata['duration']:.1f}s")
                    col2.metric("FPS", f"{metadata['fps']:.1f}")
                    col3.metric(
                        "Resolution", f"{metadata['width']}x{metadata['height']}"
                    )
                    col4.metric("Frames", metadata["total_frames"])

                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
                    return

            # Start processing
            with st.spinner("Starting processing..."):
                try:
                    process_response = start_processing(video_id)
                    st.success("✅ Processing started!")

                    # Store in session state
                    st.session_state.current_video_id = video_id

                except Exception as e:
                    st.error(f"❌ Failed to start processing: {str(e)}")
                    return

            # Progress monitoring
            progress_bar = st.progress(0)
            status_text = st.empty()

            while True:
                status = get_status(video_id)

                progress = status["progress"]
                progress_bar.progress(int(progress))
                status_text.text(
                    f"Status: {status['status']} - {progress:.1f}% - Frame {status['current_frame']}/{status['total_frames']}"
                )

                if status["status"] == "completed":
                    st.success("🎉 Processing completed!")
                    st.balloons()
                    break
                elif status["status"] == "failed":
                    st.error("❌ Processing failed!")
                    break

                time.sleep(2)


def view_results_page():
    """View processing results page"""
    st.header("📊 View Results")

    # Video ID input
    video_id = st.text_input(
        "Enter Video ID",
        value=st.session_state.get("current_video_id", ""),
        help="Enter the video ID from upload",
    )

    if not video_id:
        st.info("👆 Enter a video ID to view results")
        return

    if st.button("Load Results"):
        try:
            with st.spinner("Loading results..."):
                results = get_results(video_id)
                st.session_state.results = results
                st.success("✅ Results loaded!")
        except Exception as e:
            st.error(f"❌ Failed to load results: {str(e)}")
            return

    # Display results
    if "results" in st.session_state:
        results = st.session_state.results

        # Metadata
        st.subheader("Video Information")
        metadata = results["metadata"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration", f"{metadata['duration']:.1f}s")
        col2.metric("FPS", f"{metadata['fps']:.1f}")
        col3.metric("Resolution", f"{metadata['width']}x{metadata['height']}")
        col4.metric("Total Frames", metadata["total_frames"])

        # Summary statistics
        st.subheader("Processing Summary")
        summary = results["summary"]["video_summary"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Detections", summary["total_detections"])
        col2.metric("Unique Tracks", summary["unique_tracks"])
        col3.metric("OCR Extractions", summary["total_ocr_extractions"])
        col4.metric("Processing Time", f"{summary['processing_time']:.2f}s")

        # Download buttons
        st.subheader("📥 Downloads")
        col1, col2 = st.columns(2)

        with col1:
            # Download JSON results
            json_url = f"{API_URL}/results/{video_id}/download"
            st.markdown(f"[⬇️ Download JSON Results]({json_url})")

        with col2:
            # Download annotated video
            video_url = f"{API_URL}/results/{video_id}/video"
            st.markdown(f"[🎬 Download Annotated Video]({video_url})")
            st.caption("Video with bounding boxes for detections, tracks, and OCR")

        # Class distribution
        st.subheader("Detected Object Classes")
        class_data = summary["unique_classes"]
        if class_data:
            df = pd.DataFrame(list(class_data.items()), columns=["Class", "Count"])
            df = df.sort_values("Count", ascending=False)

            fig = px.bar(df, x="Class", y="Count", title="Object Class Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # OCR Results
        st.subheader("Extracted Text")
        unique_texts = summary.get("unique_texts", [])
        if unique_texts:
            st.write(f"Found {len(unique_texts)} unique text strings:")
            for i, text in enumerate(unique_texts[:20], 1):  # Show top 20
                st.write(f"{i}. `{text}`")
        else:
            st.info("No text detected in video")

        # Track trajectories
        st.subheader("Object Trajectories")
        track_summaries = results["summary"]["track_summaries"]

        if track_summaries:
            # Select track to visualize
            track_ids = [ts["track_id"] for ts in track_summaries]
            selected_track = st.selectbox("Select Track ID", track_ids)

            # Find selected track
            track_info = next(
                ts for ts in track_summaries if ts["track_id"] == selected_track
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Class", track_info["class_name"])
            col2.metric("Duration", f"{track_info['total_frames']} frames")
            col3.metric("First Frame", track_info["first_frame"])
            col4.metric("Last Frame", track_info["last_frame"])

            # Plot trajectory
            trajectory = track_info["trajectory"]
            if trajectory:
                x_coords = [p[0] for p in trajectory]
                y_coords = [p[1] for p in trajectory]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode="lines+markers",
                        name=f"Track {selected_track}",
                        line=dict(width=2),
                        marker=dict(size=6),
                    )
                )

                fig.update_layout(
                    title=f"Trajectory of Track {selected_track}",
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    yaxis=dict(
                        autorange="reversed"
                    ),  # Invert Y axis (image coordinates)
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

        # Frame-by-frame viewer
        st.subheader("Frame-by-Frame Viewer")
        frames = results["frames"]

        if frames:
            frame_numbers = [f["frame_number"] for f in frames]
            selected_frame_idx = st.slider(
                "Select Frame",
                0,
                len(frames) - 1,
                0,
                help="Slide to navigate through processed frames",
            )

            frame_result = frames[selected_frame_idx]

            st.write(
                f"**Frame:** {frame_result['frame_number']} | **Time:** {frame_result['timestamp']:.2f}s"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Detected Objects:**")
                if frame_result["tracks"]:
                    for track in frame_result["tracks"]:
                        st.write(
                            f"- Track {track['track_id']}: {track['class_name']} ({track['confidence']:.2f})"
                        )
                else:
                    st.info("No objects in this frame")

            with col2:
                st.write("**Extracted Text:**")
                if frame_result["ocr_results"]:
                    for ocr in frame_result["ocr_results"]:
                        st.write(f"- `{ocr['text']}` ({ocr['confidence']:.2f})")
                else:
                    st.info("No text in this frame")


def analytics_dashboard_page():
    """Analytics dashboard page"""
    st.header("📈 Analytics Dashboard")

    if "results" not in st.session_state:
        st.info("👆 Load results from 'View Results' page first")
        return

    results = st.session_state.results

    # Temporal analysis
    st.subheader("Temporal Analysis")

    frames = results["frames"]

    # Objects over time
    frame_data = []
    for frame in frames:
        frame_data.append(
            {
                "frame": frame["frame_number"],
                "timestamp": frame["timestamp"],
                "objects": len(frame["tracks"]),
                "texts": len(frame["ocr_results"]),
            }
        )

    df = pd.DataFrame(frame_data)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            df, x="timestamp", y="objects", title="Objects Detected Over Time"
        )
        fig.update_xaxis(title="Time (seconds)")
        fig.update_yaxis(title="Number of Objects")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(
            df,
            x="timestamp",
            y="texts",
            title="Text Extractions Over Time",
            color_discrete_sequence=["green"],
        )
        fig.update_xaxis(title="Time (seconds)")
        fig.update_yaxis(title="Number of Texts")
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap of activity
    st.subheader("Activity Heatmap")

    # Create bins for time segments
    bins = 10
    df["time_bin"] = pd.cut(df["timestamp"], bins=bins)
    heatmap_data = df.groupby("time_bin")["objects"].mean().reset_index()
    heatmap_data["time_range"] = heatmap_data["time_bin"].astype(str)

    fig = px.bar(
        heatmap_data,
        x="time_range",
        y="objects",
        title="Average Objects per Time Segment",
    )
    fig.update_xaxis(title="Time Range")
    fig.update_yaxis(title="Average Objects")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
