"""
Script for hosting a mm-olmo endpoint via modal.
DEPLOY:
modal deploy demo_scripts.modal_api_server
"""
import json
import time
import uuid
from threading import Thread
from typing import Any, Dict, List, Optional
from queue import Queue

import modal
import modal.gpu

N_GPU = 1

MODEL_NAME = "uber-model-v2"
# CKPT_DIR = "/net/nfs/prior/sanghol/cockatoo/models/uber-model-v9/3.1-5510-exp/step36000-unsharded"
SEQ_LEN = 1536
MAX_NEW_TOKENS = 768
BATCH_SIZE = 4
STYLE = "demo"

MODELS_DIR = "/mm-olmo"

APP_NAME = MODEL_NAME
APP_LABEL = APP_NAME.lower()


# ## Define a container image and mount the volume containing the model checkpoint
# The `HF_ACCESS_TOKEN` environment variable must be set

try:
    volume = modal.Volume.lookup("mm-olmo", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Upload checkpoint first with modal run demo_scripts.modal_upload_checkpoint --dst_dir [DIST_DIR]")


secret = modal.Secret.from_name("sanghol-mm-olmo-env")
model_image = (
    modal.Image.from_registry("sanghol/mm-olmo:latest", secret=secret)
)

app = modal.App(f"mm-olmo-{APP_NAME}")

# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `@enter` decorator.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


@app.cls(
    gpu=modal.gpu.H100(count=N_GPU),
    timeout=60 * 5,
    container_idle_timeout=60 * 5,
    keep_warm=1,
    image=model_image,
    concurrency_limit=10,
    # allow_concurrent_inputs=2,
    volumes={MODELS_DIR: volume},
    mounts=[modal.Mount.from_local_python_packages("olmo", "hf_olmo")],
)
class Model:
    @modal.enter()
    def start_api(self):
        from transformers import AutoTokenizer, TextStreamer

        from hf_olmo.modeling_olmo import MolmoForCausalLM
        from olmo.config import (
            ModelConfig,
            TokenizerConfig,
            VisionBackboneConfig,
            parse_gin_bindings,
        )

        class MMTextIteratorStreamer(TextStreamer):

            def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                self.text_queue = Queue()
                self.stop_signal = None

            def on_finalized_text(self, text: str, stream_end: bool = False):
                """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
                self.text_queue.put_nowait(text)
                if stream_end:
                    self.text_queue.put_nowait(self.stop_signal)

            def put(self, value):
                value = value[value >= 0]
                super().put(value)

            def __iter__(self):
                return self

            def __next__(self):
                value = self.text_queue.get()
                if value == self.stop_signal:
                    raise StopIteration()
                return value

        print("🥶 cold starting inference")
        start = time.monotonic_ns()

        self.seq_len = SEQ_LEN
        self.max_new_tokens = MAX_NEW_TOKENS
        self.batch_size = BATCH_SIZE
        self.style = STYLE

        self.model = MolmoForCausalLM.from_pretrained(
            MODELS_DIR + "/" + MODEL_NAME,
            return_dict=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        model_cfg = self.model.config.to_dict()
        for k in list(model_cfg):
            if k not in ModelConfig.__annotations__:
                del model_cfg[k]
        if model_cfg['vision_backbone'] is not None:
            model_cfg['vision_backbone'] = VisionBackboneConfig(**model_cfg['vision_backbone'])
        model_cfg['tokenizer'] = TokenizerConfig(**model_cfg['tokenizer'])
        self.model_cfg = ModelConfig(**model_cfg)
        if self.model_cfg.vision_backbone is not None:
            binding_list = parse_gin_bindings(self.model_cfg)
        self.preprocessor = self.model_cfg.get_preprocessor()
        self.tokenizer = self.model_cfg.get_tokenizer()

        self.streamer_cls = MMTextIteratorStreamer

        self.feature_lengths = dict(
            target_tokens=self.seq_len,
            loss_masks=self.seq_len,
            images=self.model_cfg.get_max_crops(),
            image_positions=self.model_cfg.get_max_crops(),
            image_input_idx=self.model_cfg.get_max_crops(),
            is_training=False,
        )

        duration_s = (time.monotonic_ns() - start) / 1e9
        print(f"🏎️ engine started in {duration_s:.0f}s")

    @staticmethod
    def image_to_numpy(image_str):
        import base64
        from io import BytesIO

        import numpy as np
        import requests
        from PIL import Image, ImageFile, ImageOps
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if isinstance(image_str, str) and (image_str.strip().startswith("http://") or image_str.strip().startswith("https://")):
            response = requests.get(image_str)

            # Check if the request was successful
            if response.status_code == 200:
                # Open the image from the response content
                image = Image.open(BytesIO(response.content)).convert("RGB")

                image = ImageOps.exif_transpose(image)

                # Convert the image to a NumPy array
                image_array = np.array(image).astype(np.uint8)
            else:
                raise Exception(f"Failed to download image. Status code: {response.status_code}")
        else:
            image_array = np.asarray(ImageOps.exif_transpose(Image.open(BytesIO(base64.b64decode(image_str.encode("utf-8")))).convert("RGB")))

        return image_array

    @staticmethod
    def download_image_to_numpy(url):
        from io import BytesIO

        import numpy as np
        import requests
        from PIL import Image, ImageFile, ImageOps
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Open the image from the response content
            image = Image.open(BytesIO(response.content)).convert("RGB")

            image = ImageOps.exif_transpose(image)

            # Convert the image to a NumPy array
            image_array = np.array(image).astype(np.uint8)

            return image_array
        else:
            raise Exception(f"Failed to download image. Status code: {response.status_code}")

    def get_single_example_batch(self, input_image, input_text):
        import functools

        import numpy as np
        import tensorflow as tf
        import torch

        image_array = self.image_to_numpy(input_image)
        # FIXME: We need to modify the demo code so as to pass an empty string for captioner models
        input_example = dict(prompt=input_text if input_text is not None else "")
        # input_example = dict(prompt="")
        # preprocess_fn = tf.function(
        #     functools.partial(self.preprocessor.preprocess, is_training=False, for_inference=True, pad_images=False, style=self.style),
        # )
        # batch = preprocess_fn(image_array, input_example)
        batch = self.preprocessor.preprocess(
            image_array,
            input_example,
            is_training=False,
            seq_len=self.feature_lengths["target_tokens"],
            pad_images=True,
            style=self.style,
        )
        batch = {k: np.expand_dims(v.numpy(), 0) for k, v in batch.items()}
        batch = {k: torch.from_numpy(v.copy()) if not k.startswith("metadata/") else v.copy() for k, v in batch.items()}
        return batch

    def get_batch_ds(self, examples):
        from dataclasses import replace

        import tensorflow as tf

        from olmo.mm_data.iterable_dataset import PyTorchDatasetIterator

        new_examples = []
        for example in examples:
            new_ex = {}
            for k, v in example.items():
                if k == "input_image":
                    new_ex["image"] = self.image_to_numpy(v)
                elif k == "prompt" and example[k]:
                    # FIXME: We need to modify the demo code so as to pass an empty string for captioner models
                    # new_ex["prompt"] = ""
                    new_ex["prompt"] = v
                elif k == "example_id":
                    new_ex["metadata/example_id"] = v
                elif k == "length_cond" and example[k]:
                    new_ex["length_cond"] = v
                else:
                    new_ex[k] = v
            if "style" not in new_ex:
                # FIXME can we avoid this default? It might cause unexpected behavior
                new_ex["style"] = self.style
            new_examples.append(new_ex)

        dataset = tf.data.experimental.from_list(new_examples)
        preprocess_fn = replace(self.preprocessor).get_preprocessor(is_training=False, for_inference=True)
        dataset = preprocess_fn(dataset)

        converter = self.preprocessor.get_post_mixing_preprocessor()
        dataset = converter(dataset, task_feature_lengths=self.feature_lengths)

        data_iter = dataset.batch(self.batch_size, drop_remainder=False)
        data_iter = PyTorchDatasetIterator(data_iter, checkpoint=False, for_inference=True)
        return data_iter

    @modal.method()
    async def batch_generate(self, input_image: List[str], prompt: List[str]):
        import torch
        from transformers import GenerationConfig

        from olmo.torch_util import move_to_device

        # Set CUDA device.
        device = torch.device("cuda")

        generation_config = GenerationConfig(
            temperature=None,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=-1,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            top_p=None,
        )

        examples = [
            {"input_image": img, "prompt": txt}
            for img, txt in zip(input_image, prompt)
        ]
        start_data = time.monotonic_ns()
        is_singleton = len(examples) == 1
        if is_singleton:
            batch = self.get_single_example_batch(input_image[0], prompt[0])
            data_iter = [batch]
        else:
            data_iter = self.get_batch_ds(examples)
        elapsed_data_s = (time.monotonic_ns() - start_data) / 1e9

        all_predictions = []
        start_infer = time.monotonic_ns()
        for batch in data_iter:
            batch_inference = {k: v for k, v in batch.items() if not k.startswith("metadata/")}
            batch_inference["input_ids"] = batch_inference.pop("input_tokens")
            batch_inference = move_to_device(batch_inference, device)
            seq_len = batch_inference["input_ids"].shape[1]

            with torch.inference_mode():
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    olmo_gen_output = self.model.generate_from_batch(
                        batch_inference,
                        generation_config,
                    )

            predictions = olmo_gen_output.detach().cpu().numpy()
            all_predictions += [self.tokenizer.decode(seq[seq_len:]) for seq in predictions]
        elapsed_infer_s = (time.monotonic_ns() - start_infer) / 1e9
        return all_predictions, elapsed_data_s, elapsed_infer_s

    @modal.method()
    async def completion_stream(self, input_image: str, prompt: str, opts: dict):
        import torch
        from transformers import GenerationConfig, TextIteratorStreamer

        from olmo.torch_util import move_to_device

        # Set CUDA device.
        device = torch.device("cuda")

        max_new_tokens = int(opts.get("max_new_tokens", self.max_new_tokens))

        generation_config = GenerationConfig(
            temperature=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=-1,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            top_p=None,
        )

        request_id = random_uuid()
        batch = self.get_single_example_batch(input_image, prompt)
        batch["input_ids"] = batch.pop("input_tokens")
        batch = move_to_device(batch, device)

        def autocast_generate_from_batch(
            batch: Dict[str, Any],
            generation_config: Optional[GenerationConfig] = None,
            **kwargs,
        ):
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                out = self.model.generate_from_batch(batch, generation_config=generation_config, **kwargs)
            return out

        start = time.monotonic_ns()
        streamer = self.streamer_cls(self.tokenizer, skip_prompt=True)
        thread = Thread(target=autocast_generate_from_batch, kwargs=dict(batch=batch, generation_config=generation_config, streamer=streamer))
        thread.start()

        for next_text in streamer:
            elapsed_s = (time.monotonic_ns() - start) / 1e9
            yield next_text, request_id, elapsed_s

        thread.join()

    @modal.method()
    async def completion(self, input_image: str, prompt: str, opts: dict):
        import torch
        from transformers import GenerationConfig

        from olmo.torch_util import move_to_device

        # Set CUDA device.
        device = torch.device("cuda")

        max_new_tokens = int(opts.get("max_new_tokens", self.max_new_tokens))

        generation_config = GenerationConfig(
            temperature=None,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=-1,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            top_p=None,
        )

        request_id = random_uuid()
        start_data = time.monotonic_ns()
        batch = self.get_single_example_batch(input_image, prompt)
        elapsed_data_s = (time.monotonic_ns() - start_data) / 1e9
        batch["input_ids"] = batch.pop("input_tokens")
        seq_len = batch["input_ids"].shape[1]
        batch = move_to_device(batch, device)

        start_infer = time.monotonic_ns()
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                olmo_gen_output = self.model.generate_from_batch(
                    batch,
                    generation_config,
                )
        text = self.tokenizer.decode(olmo_gen_output.detach().cpu().numpy()[0][seq_len:])
        elapsed_infer_s = (time.monotonic_ns() - start_infer) / 1e9
        return text, request_id, elapsed_data_s, elapsed_infer_s


# ## Coupling a frontend web application
#
# We can stream inference from a FastAPI backend, also deployed on Modal.

from modal import asgi_app

api_image = (
    modal.Image.debian_slim(python_version="3.11")
)

@app.function(
    keep_warm=1,
    allow_concurrent_inputs=5,
    timeout=60 * 10,
)
@modal.batched(max_batch_size=4, wait_ms=100)
async def generate(inputs: List[dict]):
    from fastapi.responses import JSONResponse
    responses = []
    try:
        input_image = [inp["image"] for inp in inputs]
        prompt = [inp["prompt"] for inp in inputs]
    except ValueError as e:
        return [JSONResponse({"finish_reason": "error", "text": ""}) for _ in inputs]
    model = Model()
    all_predictions, data_s, infer_s = await model.batch_generate.remote.aio(input_image, prompt)
    for text in all_predictions:
        request_id = random_uuid()
        response = [{"example_id": request_id, "prediction": text}]
        # output = dict(text=text)
        # result = dict(output=output, dataTime=f"{data_s}s", inferenceTime=f"{infer_s}s")
        # response = dict(requestId=request_id, result=result)
        responses.append(JSONResponse(response))

    return responses

@app.function(
    image=api_image,
    keep_warm=1,
    allow_concurrent_inputs=20,
    timeout=60 * 10,
)
@asgi_app(label=APP_LABEL)
def model_web():
    import fastapi
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse

    web_app = fastapi.FastAPI()

    # Add CORSMiddleware to the application
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    @web_app.post("/completion")
    async def completion(inp: dict, user_agent: Optional[str] = fastapi.Header(None)):
        try:
            input_image = inp["input_image"][0]
            prompt = inp["input_text"][0]
            opts = inp.get("opts", {})
        except ValueError as e:
            return JSONResponse({"finish_reason": "error", "text": ""})
        model = Model()
        text, request_id, data_s, infer_s = await model.completion.remote.aio(input_image, prompt, opts)
        """
        output = dict(text=text)
        result = dict(output=output, dataTime=f"{data_s}s", inferenceTime=f"{infer_s}s")
        response = dict(requestId=request_id, result=result)
        """
        response = [{"example_id": request_id, "prediction": text}]

        return JSONResponse(response)

    @web_app.post("/batch_generate")
    async def batch_generate(inp: dict):
        inp = {"image": inp["input_image"][0], "prompt": inp["input_text"][0]}
        response = await generate.remote.aio(inp)
        return response

    @web_app.post("/completion_stream")
    async def completion_stream(inp: dict, user_agent: Optional[str] = fastapi.Header(None)):
        async def generate():
            try:
                input_image = inp["input_image"][0]
                prompt = inp["input_text"][0]
                opts = inp.get("opts", {})
            except ValueError as e:
                yield f"{json.dumps({'finish_reason': 'error', 'text': ''})}\n".encode("utf-8")
                return
            model = Model()
            async for text, request_id, infer_s in model.completion_stream.remote_gen.aio(
                input_image, prompt, opts
            ):

                output = dict(text=text)
                result = dict(output=output, inferenceTime=f"{infer_s}s")
                response = dict(requestId=request_id, result=result)

                yield f"{json.dumps(response, ensure_ascii=False)}\n".encode("utf-8")

        return StreamingResponse(generate(), media_type="text/event-stream")

    return web_app


# ## Coupling a deployed function endpoint
#
# See also https://modal.com/docs/guide/trigger-deployed-functions for
# for client-side use

@app.function(
    image=api_image,
    keep_warm=1,
    allow_concurrent_inputs=20,
    timeout=60 * 10,
)
async def model_api(inp: dict):
    try:
        input_image = inp["image"]
        prompt = inp["prompt"]
        opts = inp.get("opts", {})
    except ValueError as e:
        return {"finish_reason": "error", "text": ""}

    model = Model()
    text, request_id, data_s, infer_s = await model.completion.remote.aio(input_image, prompt, opts)
    output = dict(text=text)
    result = dict(output=output, dataTime=f"{data_s}s", inferenceTime=f"{infer_s}s")
    response = dict(requestId=request_id, result=result)
    return response

# This local entry point allows you to test inference without deploying the Modal app.
# It should be used by running the following command from the repository root.
# > modal run demo_scripts/modal_api_server.py::app.main
@app.local_entrypoint()
async def main():
    print("=== local entrypoint")
    print("=== lazily initializing model")
    model = Model()
    print("=== completion_stream start")
    result = model.completion_stream.remote_gen(
            "https://www.datocms-assets.com/64837/1721697383-wildlands-trees.jpg",
            "What does the image depict?",
        {},
        )
    for res in result:
        print(res)
    print("=== completion_stream complete")

