import os, sys, importlib.util, shutil
import modal

# --------------------------------------
# SAM2 on Modal • Scheme A (runtime checkpoints + volumes)
# - Checkpoints downloaded at runtime into a Modal Volume
# - Frontend serves static bundle and proxies /api/* -> backend/*
# - Backend imported from /opt/sam2/server/app.py (Flask WSGI wrapped to ASGI)
# --------------------------------------

app = modal.App("sam2-modal")

# ------------------ Volumes ------------------
sam2_data = modal.Volume.from_name("sam2-data", create_if_missing=True)
sam2_ckpts = modal.Volume.from_name("sam2-ckpts", create_if_missing=True)

# ------------------ Optional: seed ./demo/data into /data ------------------
seed_image = modal.Image.debian_slim().add_local_dir("./demo/data", "/seed", copy=True)

@app.function(image=seed_image, volumes={"/data": sam2_data})
def seed_data():
    os.makedirs("/data", exist_ok=True)
    for root, _, files in os.walk("/seed"):
        rel = os.path.relpath(root, "/seed")
        dst = os.path.join("/data", rel) if rel != "." else "/data"
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(root, f), os.path.join(dst, f))
    return "Seeded /data"

# ------------------ Checkpoint downloader (lightweight image) ------------------
downloader_image = (
    modal.Image.debian_slim()
    .apt_install("curl", "ca-certificates")
    .pip_install("requests")
)

CKPTS = {
    "sam2.1_hiera_tiny.pt":      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt":     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt":     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
}

@app.function(image=downloader_image, volumes={"/opt/sam2/checkpoints": sam2_ckpts})
def fetch_checkpoints():
    import requests
    os.makedirs("/opt/sam2/checkpoints", exist_ok=True)
    results = {}
    for name, url in CKPTS.items():
        out = f"/opt/sam2/checkpoints/{name}"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            results[name] = "exists"
            continue
        tmp = out + ".part"
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp, out)
        results[name] = f"downloaded ({os.path.getsize(out)} bytes)"
    return results

@app.function(volumes={"/opt/sam2/checkpoints": sam2_ckpts})
def ls_ckpts():
    root = "/opt/sam2/checkpoints"
    if not os.path.exists(root):
        return {"exists": False, "files": []}
    files = sorted(os.listdir(root))
    sizes = {f: os.path.getsize(os.path.join(root, f)) for f in files}
    return {"exists": True, "files": files, "sizes": sizes}

@app.function(volumes={"/opt/sam2/checkpoints": sam2_ckpts})
def stat_file(name: str):
    p = os.path.join("/opt/sam2/checkpoints", name)
    return {"exists": os.path.exists(p), "size": os.path.getsize(p) if os.path.exists(p) else 0}

# ------------------ Backend image from your Dockerfile ------------------
backend_image = (
    modal.Image.from_dockerfile(
        path="backend.Dockerfile",
        context_dir=".",
    )
    # We wrap Flask (WSGI) with Starlette's WSGIMiddleware -> ensure starlette is available
    .pip_install("starlette")
    .env({
        "SERVER_ENVIRONMENT": "DEV",
        "GUNICORN_WORKERS": "1",
        "GUNICORN_THREADS": "2",
        "GUNICORN_PORT": "5000",
        "DEFAULT_VIDEO_PATH": "gallery/05_default_juggle.mp4",
        "FFMPEG_NUM_THREADS": "1",
        "VIDEO_ENCODE_CODEC": "libx264",
        "VIDEO_ENCODE_CRF": "23",
        "VIDEO_ENCODE_FPS": "24",
        "VIDEO_ENCODE_MAX_WIDTH": "1280",
        "VIDEO_ENCODE_MAX_HEIGHT": "720",
        "VIDEO_ENCODE_VERBOSE": "False",
        # "API_URL": "https://uwgzq--sam2-modal-backend-dev.modal.run",  # Set before import
    })
)

# ------------------ Backend web endpoint (wrap Flask -> ASGI) ------------------
@app.function(
    image=backend_image,
    gpu="A100",
    volumes={"/data": sam2_data, "/opt/sam2/checkpoints": sam2_ckpts},
    min_containers=1,      
    max_containers = 1,
    concurrency_limit=1, 
    scaledown_window=900,   
)
@modal.asgi_app()
def backend():
    # Ensure model size default and set real API_URL BEFORE importing backend app
    os.environ.setdefault("MODEL_SIZE", "base_plus")
    os.environ["API_URL"] = backend.get_web_url()

    # Guard: ensure checkpoint exists
    expected = "/opt/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    if not (os.path.exists(expected) and os.path.getsize(expected) > 0):
        from fastapi import FastAPI
        api = FastAPI(title="sam2-backend-missing-ckpt")
        @api.get("/healthz")
        def healthz():
            return {
                "ok": False,
                "error": f"checkpoint not found: {expected}",
                "hint": "Run `modal run sam2_modal_app.py::fetch_checkpoints` and redeploy",
            }
        return api

    code_root = "/opt/sam2/server"
    if code_root not in sys.path:
        sys.path.insert(0, code_root)

    try:
        # Import user's Flask app from /opt/sam2/server/app.py
        spec = importlib.util.spec_from_file_location("sam2_backend_app", os.path.join(code_root, "app.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        app_obj = getattr(module, "app", None)
        if app_obj is None:
            raise RuntimeError("/opt/sam2/server/app.py does not define `app`")

        # If it's Flask/WSGI, wrap to ASGI; otherwise return as-is (e.g., FastAPI)
        try:
            from flask import Flask  # type: ignore
        except Exception:
            Flask = None  # type: ignore
        if Flask is not None and isinstance(app_obj, Flask):
            from starlette.middleware.wsgi import WSGIMiddleware
            return WSGIMiddleware(app_obj)
        else:
            return app_obj

    except Exception as e:
        from fastapi import FastAPI
        api = FastAPI(title="sam2-backend-fallback")
        @api.get("/healthz")
        def healthz():
            return {"ok": False, "error": str(e), "data_mounted": os.path.exists("/data")}
        return api

# ------------------ Frontend (static + proxy) ------------------
frontend_image = (
    modal.Image.from_dockerfile(
        path="demo/frontend/frontend.Dockerfile",
        context_dir="demo/frontend",
        add_python="3.11",
    )
    .pip_install("fastapi", "uvicorn", "starlette", "httpx")
)

@app.function(image=frontend_image)
@modal.asgi_app()
def frontend():
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles
    from starlette.responses import JSONResponse, Response
    from starlette.middleware.cors import CORSMiddleware
    import httpx, os

    STATIC_DIR = "/usr/share/nginx/html"
    BACKEND_URL = os.environ.get("BACKEND_URL") or backend.get_web_url()

    async def proxy(request):
        if request.method.upper() == "OPTIONS":
            return Response(status_code=204)

        BACKEND_URL = os.environ.get("BACKEND_URL") or backend.get_web_url()
        # /api/* -> backend /*
        path_in = request.url.path.removeprefix("/api")
        url = BACKEND_URL + path_in

        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        headers["x-forwarded-proto"] = "https"
        headers["x-forwarded-host"] = request.url.hostname or "frontend"
        if path_in.endswith("/graphql"):
            headers["accept"] = "application/json"
            headers.setdefault("content-type", "application/json")

        body = await request.body()
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.request(
                request.method, url, content=body, headers=headers, params=request.query_params
            )

        hop_by_hop = {
            "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
            "te", "trailers", "transfer-encoding", "upgrade", "content-length",
        }
        resp_headers = {}
        for k, v in r.headers.items():
            kl = k.lower()
            if kl in hop_by_hop or kl.startswith("modal-"):
                continue
            if kl != "set-cookie":
                resp_headers[k] = v

        try:
            raw_cookies = r.headers.get_list("set-cookie")  # httpx 多值头
        except Exception:
            raw_cookies = [v for k, v in r.headers.items() if k.lower() == "set-cookie"]

        frontend_host = request.url.hostname or ""

        def rewrite_cookie(val: str, new_host: str) -> str:
            parts = [p.strip() for p in val.split(";")]
            out, seen_domain, seen_path = [], False, False
            low_all = val.lower()
            for p in parts:
                pl = p.lower()
                if pl.startswith("domain="):
                    out.append(f"Domain={new_host}")
                    seen_domain = True
                elif pl.startswith("path="):
                    out.append("Path=/api") 
                    seen_path = True
                else:
                    out.append(p)
            if not seen_domain:
                out.append(f"Domain={new_host}")
            if not seen_path:
                out.append("Path=/api")
            if "samesite" not in low_all:
                out.append("SameSite=None")
            if "secure" not in low_all:
                out.append("Secure")
            return "; ".join(out)

        resp = Response(r.content, status_code=r.status_code, headers=resp_headers)
        for c in raw_cookies:
            resp.headers.append("set-cookie", rewrite_cookie(c, frontend_host))
        return resp


    app = Starlette(routes=[
        # IMPORTANT: allow POST for GraphQL
        Route("/api/{path:path}", proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]),
        Route("/healthz", lambda req: JSONResponse({"ok": True, "backend": BACKEND_URL}), methods=["GET"]),
        Mount("/", app=StaticFiles(directory=STATIC_DIR, html=True), name="static"),
    ])

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

# ------------------ Utilities ------------------
@app.function()
def print_urls():
    return {"frontend": frontend.get_web_url(), "backend": backend.get_web_url()}