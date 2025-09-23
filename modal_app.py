"""Modal deployment entrypoint for the SAM 2 demo."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import modal

APP_NAME = "sam2-demo"

app = modal.App(APP_NAME)

backend_image = modal.Image.from_dockerfile(
    context=".",
    dockerfile="backend.Dockerfile",
)


@app.function(
    image=backend_image,
    gpu=modal.gpu.A10G(),
    timeout=60 * 60,
    container_idle_timeout=60 * 15,
    allow_concurrent_inputs=8,
    env={
        "SERVER_ENVIRONMENT": "MODAL",
        "DATA_PATH": "/data",
        "DEFAULT_VIDEO_PATH": "gallery/05_default_juggle.mp4",
        "FRONTEND_DIST_PATH": "/opt/sam2/server/frontend_dist",
    },
)
@modal.wsgi_app()
def sam2_demo():
    """Expose the Flask app and ensure demo assets are writable."""
    data_path = Path(os.environ["DATA_PATH"])
    data_path.mkdir(parents=True, exist_ok=True)

    seed_path = Path("/opt/sam2/server/data")
    for child in seed_path.iterdir():
        destination = data_path / child.name
        if destination.exists():
            continue
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)

    from app import app as flask_app

    return flask_app


@app.local_entrypoint()
def deploy():
    """Deploy the Modal app and print the public URL."""
    deployment = app.deploy()
    print("SAM 2 demo backend deployed.")
    for function in deployment.web_endpoints:
        if function.function_name == "sam2_demo":
            print(f"Public URL: {function.url}")
            break

