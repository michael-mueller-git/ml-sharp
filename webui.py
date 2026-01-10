"""WebUI for ml-sharp 3D Gaussian Splat prediction.

A simple Flask-based web interface for uploading images and generating 3DGS PLY files.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request, send_file

from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Flask app - use absolute paths for static and template folders
_base_dir = Path(__file__).parent.absolute()
app = Flask(
    __name__,
    static_folder=str(_base_dir / "webui_static"),
    static_url_path="/static",
    template_folder=str(_base_dir / "webui_templates")
)

# Global model cache
_model_cache = {"predictor": None, "device": None}

# Output directory for generated PLY files
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model URL
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_predictor() -> tuple[RGBGaussianPredictor, torch.device]:
    """Get or create the Gaussian predictor model."""
    if _model_cache["predictor"] is None:
        device = get_device()
        LOGGER.info(f"Loading model on device: {device}")

        # Download and load model
        LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)

        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        predictor.to(device)

        _model_cache["predictor"] = predictor
        _model_cache["device"] = device
        LOGGER.info("Model loaded successfully")

    return _model_cache["predictor"], _model_cache["device"]


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/test")
def test_viewer():
    """Serve the test viewer page."""
    return render_template("test-viewer.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate a 3DGS PLY file from an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Check file extension
    allowed_extensions = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".tiff", ".tif", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        # Save uploaded file temporarily
        unique_id = str(uuid.uuid4())[:8]
        original_stem = Path(file.filename).stem

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        LOGGER.info(f"Processing uploaded file: {file.filename}")

        # Load the image
        image, _, f_px = io.load_rgb(tmp_path)
        height, width = image.shape[:2]

        # Get the model
        predictor, device = get_predictor()

        # Run prediction
        gaussians = predict_image(predictor, image, f_px, device)

        # Save the PLY file
        output_filename = f"{original_stem}_{unique_id}.ply"
        output_path = OUTPUT_DIR / output_filename
        save_ply(gaussians, f_px, (height, width), output_path)

        LOGGER.info(f"Saved PLY to: {output_path}")

        # Clean up temp file
        tmp_path.unlink()

        return jsonify({
            "success": True,
            "filename": output_filename,
            "download_url": f"/download/{output_filename}",
            "view_url": f"/ply/{output_filename}",
        })

    except Exception as e:
        LOGGER.exception("Error during generation")
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download(filename: str):
    """Download a generated PLY file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype="application/octet-stream",
    )


@app.route("/ply/<filename>")
def serve_ply(filename: str):
    """Serve a PLY file for the viewer."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(
        file_path,
        mimetype="application/octet-stream",
    )


@app.route("/status")
def status():
    """Get server status."""
    device = get_device()
    model_loaded = _model_cache["predictor"] is not None
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model_loaded": model_loaded,
        "cuda_available": torch.cuda.is_available(),
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ml-sharp WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")

    args = parser.parse_args()

    if args.preload:
        LOGGER.info("Preloading model...")
        get_predictor()

    LOGGER.info(f"Starting WebUI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
