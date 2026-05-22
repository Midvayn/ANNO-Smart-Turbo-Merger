import os
import json
import traceback
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from safetensors.torch import load_file, save_file
import uvicorn

APP_NAME = "ANNO Smart Turbo Merger"
BASE_DIR = "models"

app = FastAPI(title=APP_NAME)


def clean_key(key: str) -> str:
    prefixes = (
        "model.diffusion_model.",
        "diffusion_model.",
        "first_stage_model.",
    )

    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]

    return key


def safe_output_name(name: str) -> str:
    cleaned = "".join(c for c in name.strip() if c.isalnum() or c in ("-", "_", ".", " "))
    cleaned = cleaned.strip().replace(" ", "_")
    return cleaned or "merged_model"


@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html was not found.</h1>", status_code=404)


@app.get("/list")
async def list_files(type: str):
    if type not in ("loras", "checkpoints"):
        return []

    path = os.path.join(BASE_DIR, type)
    os.makedirs(path, exist_ok=True)

    return sorted(
        file for file in os.listdir(path)
        if file.lower().endswith(".safetensors")
    )


@app.post("/merge")
async def merge(data: dict):
    try:
        model_type = data.get("type", "loras")
        if model_type not in ("loras", "checkpoints"):
            return {"error": "Unsupported model type."}

        file_a = data.get("file_a", "")
        file_b = data.get("file_b", "")
        if not file_a or not file_b:
            return {"error": "Please select both input files."}

        path_a = os.path.join(BASE_DIR, model_type, file_a)
        path_b = os.path.join(BASE_DIR, model_type, file_b)

        if not os.path.exists(path_a):
            return {"error": f"File A was not found: {file_a}"}
        if not os.path.exists(path_b):
            return {"error": f"File B was not found: {file_b}"}

        alpha = float(data.get("alpha", 0.5))
        alpha = max(0.0, min(1.0, alpha))

        print("\n" + "=" * 70)
        print(f"{APP_NAME} started")
        print("=" * 70)
        print(f"Model type: {model_type}")
        print(f"File A: {file_a}")
        print(f"File B: {file_b}")
        print(f"Alpha: {alpha:.2f}")

        weights_a = load_file(path_a)
        weights_b = load_file(path_b)

        if not weights_a:
            return {"error": "File A has no tensors."}
        if not weights_b:
            return {"error": "File B has no tensors."}

        target_dtype = next(iter(weights_a.values())).dtype
        print(f"Output dtype: {target_dtype}")

        map_b = {clean_key(key): key for key in weights_b.keys()}

        merged = {}
        matched_count = 0
        shape_mismatch_count = 0
        unmatched_count = 0

        for key_a, tensor_a_original in weights_a.items():
            clean = clean_key(key_a)

            if clean not in map_b:
                merged[key_a] = tensor_a_original
                unmatched_count += 1
                continue

            key_b = map_b[clean]
            tensor_b_original = weights_b[key_b]

            if tensor_a_original.shape != tensor_b_original.shape:
                merged[key_a] = tensor_a_original
                shape_mismatch_count += 1
                continue

            tensor_a = tensor_a_original.to(torch.float32)
            tensor_b = tensor_b_original.to(torch.float32)
            merged_tensor = (1.0 - alpha) * tensor_a + alpha * tensor_b

            merged[key_a] = merged_tensor.to(target_dtype)
            matched_count += 1

            if matched_count % 100 == 0:
                print(f"Processed {matched_count} matched tensors...")

        if matched_count == 0:
            return {
                "error": "No matching tensors were found. The files are probably incompatible."
            }

        output_name = safe_output_name(data.get("output", ""))
        if not output_name:
            output_name = f"merged_a{int((1 - alpha) * 100)}_b{int(alpha * 100)}"

        if not output_name.lower().endswith(".safetensors"):
            output_name += ".safetensors"

        output_path = os.path.join(BASE_DIR, model_type, output_name)

        save_file(merged, output_path)

        print("=" * 70)
        print("Merge complete")
        print(f"Matched tensors: {matched_count}")
        print(f"Unmatched tensors kept from A: {unmatched_count}")
        print(f"Shape mismatches kept from A: {shape_mismatch_count}")
        print(f"Saved as: {output_path}")
        print("=" * 70 + "\n")

        return {
            "status": (
                f"Done. Matched tensors: {matched_count}. "
                f"Unmatched kept from A: {unmatched_count}. "
                f"Shape mismatches kept from A: {shape_mismatch_count}. "
                f"Saved as: {output_name}"
            ),
            "file": output_name,
            "matched": matched_count,
            "unmatched": unmatched_count,
            "shape_mismatches": shape_mismatch_count
        }

    except Exception as error:
        print("\nCritical error:")
        traceback.print_exc()
        return {"error": str(error)}


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "loras"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  {APP_NAME}")
    print("=" * 70)
    print(f"  Models folder: {os.path.abspath(BASE_DIR)}")
    print("  LoRA folder: models/loras")
    print("  Checkpoint folder: models/checkpoints")
    print("  Address: http://127.0.0.1:7860")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=7860)
