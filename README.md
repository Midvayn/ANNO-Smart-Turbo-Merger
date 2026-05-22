# ANNO Smart Turbo Merger

ANNO Smart Turbo Merger is a small local web tool for merging `.safetensors` LoRA files and checkpoint files.

It is designed for AI model creators who want a simple local interface for weighted-sum merging without uploading models to any external service.

## Screenshots

![ANNO-Smart-Turbo-Manager UI](screenshots/Screenshot1.jpeg)
## Features

- Merge `.safetensors` LoRA files
- Merge `.safetensors` checkpoint files
- Weighted Sum blending
- Alpha slider for donor strength
- Smart key cleanup for common model prefixes
- Preserves the dtype of File A
- Keeps unmatched tensors from File A
- Saves the merged result into the selected model folder
- Local FastAPI web interface

## Project Structure

```text
ANNO-Smart-Turbo-Merger/
├─ ANNO_Smart_Turbo_Merger.py
├─ index.html
├─ Run_ANNO_Smart_Turbo_Merger.bat
├─ requirements.txt
├─ README.md
├─ LICENSE.txt
├─ .gitignore
└─ models/
   ├─ loras/
   │  └─ .gitkeep
   └─ checkpoints/
      └─ .gitkeep
```

## Installation

Install Python 3.10 or newer.

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows, you can also run:

```bat
Run_ANNO_Smart_Turbo_Merger.bat
```

## Folder Setup

Put LoRA files here:

```text
models/loras/
```

Put checkpoint files here:

```text
models/checkpoints/
```

Only `.safetensors` files are listed by the interface.

## Running

Start the app:

```bat
Run_ANNO_Smart_Turbo_Merger.bat
```

or:

```bash
python ANNO_Smart_Turbo_Merger.py
```

Then open this address in your browser:

```text
http://127.0.0.1:7860
```

## Basic Workflow

1. Put your `.safetensors` files into `models/loras` or `models/checkpoints`.
2. Start the app.
3. Select LoRA or Checkpoints.
4. Select File A as the base.
5. Select File B as the donor.
6. Set Alpha.
7. Enter an output filename.
8. Click **Start merge**.
9. The merged file is saved into the same selected folder.

## Notes

- The app runs locally.
- Large checkpoints can require a lot of RAM and VRAM/system memory.
- File A defines the output tensor dtype.
- Tensors that do not match are kept from File A.
- This tool does not download models.
- Keep backups of important models before experimenting.

## Credits

Powered by ChatGPT and ANNO.
