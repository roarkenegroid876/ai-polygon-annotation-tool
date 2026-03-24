---
title: AI Polygon Annotation Tool
emoji: 🔷
colorFrom: blue
colorTo: purple
sdk: docker
app_file: main.py
pinned: false
---

<div align="center">

<img src="https://img.shields.io/badge/version-2.0-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/python-3.10-blue?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />

<br/><br/>

# PolyAnnot v2.0

### AI-Powered Automatic Image Annotation Tool

**No manual labeling. No expensive software. Just upload and annotate.**

<br/>

[![Live Demo](https://img.shields.io/badge/Try%20Live%20Demo-HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://0k1nx0-ai-polygon-annotation-tool.hf.space)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/0k1nx0/ai-polygon-annotation-tool)

<br/>

</div>

---

## What is PolyAnnot?

**PolyAnnot** is a free, open-source web app that automatically annotates your images using AI — no drawing, no clicking, no manual work.

You upload an image. The AI detects every object, draws precise polygon outlines around them, and exports everything as a **COCO JSON dataset** — ready to use for training your own AI models.

It uses two of the most powerful open-source AI models:
- **YOLOv8x** by Ultralytics — detects what objects are in the image
- **SAM ViT-H** by Meta AI — draws pixel-perfect masks around each object

> Perfect for students, researchers, and developers building computer vision projects.

---

## Features

| Feature | Description |
|---|---|
| 🤖 Auto Annotation | Detects and annotates objects with zero manual effort |
| 🔷 Polygon Mode | Pixel-perfect outlines using YOLOv8 + SAM |
| ⬜ Bounding Box Mode | Fast rectangular annotations using YOLOv8 only |
| ✏️ Correction Tools | Drag polygon points, resize bounding boxes |
| 👍 Review System | Thumbs up / down feedback per annotation |
| ↩️ Undo / Redo | Full history with Ctrl+Z and Ctrl+Y |
| 🎨 Multi-class Colors | Each detected class gets a unique color |
| 🔍 Zoom & Pan | Inspect large images comfortably |
| 📦 COCO JSON Export | Standard format compatible with all platforms |
| 🗄️ Feedback Database | All corrections saved locally in SQLite |
| ☁️ Auto Cloud Backup | Feedback synced to HuggingFace Dataset repo |

---

## Live Demo

**[Try it now on Hugging Face Spaces](https://0k1nx0-ai-polygon-annotation-tool.hf.space)**

Upload any image and get instant AI annotations — no setup, no install, runs in your browser.

---

## How It Works

```
1. Upload Image
       │
       ▼
2. YOLOv8x Detection
   → Finds all objects in the image
   → Returns bounding boxes + class labels + confidence scores
       │
       ▼  (Polygon mode only)
3. SAM ViT-H Segmentation
   → Uses YOLO boxes as input prompts
   → Returns pixel-level masks for each object
       │
       ▼
4. OpenCV Contour Extraction
   → Converts pixel masks into polygon coordinates
       │
       ▼
5. Review & Correct  (optional)
   → User edits points, gives thumbs up/down feedback
       │
       ▼
6. Export COCO JSON
   → Ready-to-use training dataset
```

---

## Annotation Modes

| | Polygon Mode | Bounding Box Mode |
|---|---|---|
| **Models Used** | YOLOv8 + SAM | YOLOv8 only |
| **Speed** | 15 – 60 seconds | 3 – 15 seconds |
| **Precision** | Pixel-perfect outline | Object-level rectangle |
| **Best For** | Segmentation training | Detection training |

---

## Local Setup

> **Requirements:** Python 3.9+, Git, ~4 GB free disk space

**Step 1 — Clone the repository**
```bash
git clone https://github.com/0k1nx0/ai-polygon-annotation-tool.git
cd ai-polygon-annotation-tool
```

**Step 2 — Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**Step 4 — Run the server**
```bash
uvicorn main:app --reload
```

**Step 5 — Open in your browser**
```
http://127.0.0.1:8000
```

> On first run, YOLOv8 (~137 MB) and SAM (~2.5 GB) weights are downloaded automatically. This takes a few minutes on first launch only.

---

## Project Structure

```
ai-polygon-annotation-tool/
│
├── static/
│   └── index.html           # Frontend (Canvas UI, controls)
├── main.py                  # Backend (FastAPI, AI models, feedback API)
├── Dockerfile               # Container config for HuggingFace
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .gitignore
```

Generated automatically at runtime:
```
uploads/                     # Temporary uploaded images
dataset/
  ├── images/                # Saved copies of annotated images
  └── annotations/
      └── dataset.json       # COCO JSON output
feedback/
  └── feedback.db            # SQLite feedback database
yolov8x-seg.pt               # Auto-downloaded on first run (~137 MB)
sam_vit_h_4b8939.pth         # Auto-downloaded on first run (~2.5 GB)
```

---

## Output Format — COCO JSON

```json
{
  "images": [
    { "id": 1, "file_name": "photo.jpg", "width": 1280, "height": 720 }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "category_name": "person",
      "segmentation": [[120, 80, 135, 95, 150, 110]],
      "area": 4820.5,
      "bbox": [120, 80, 200, 300],
      "iscrowd": 0
    }
  ],
  "categories": [
    { "id": 1, "name": "person" }
  ]
}
```

---

## Platform Compatibility

The COCO JSON output works directly with every major ML platform — no conversion needed.

| Platform | How to Import |
|---|---|
| **Roboflow** | Upload → Select COCO JSON |
| **CVAT** | Projects → Create Task → Upload COCO JSON |
| **Detectron2** | Native COCO JSON support |
| **MMDetection** | Native COCO JSON support |
| **YOLOv8 Training** | Convert via `ultralytics` COCO to YOLO utility |
| **Hugging Face Datasets** | Load directly with the `datasets` library |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Object Detection | YOLOv8x (Ultralytics) |
| Segmentation | SAM ViT-H (Meta AI) |
| Image Processing | OpenCV + NumPy |
| Database | SQLite |
| Frontend | HTML5 + Canvas API |
| Deployment | Docker on HuggingFace Spaces |

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.10 |
| RAM | 8 GB | 16 GB+ |
| Disk | 4 GB | 10 GB+ |
| GPU | Not required | NVIDIA CUDA (much faster) |
| OS | Windows / Mac / Linux | Any |

---

## Roadmap

- [x] YOLOv8 + SAM polygon annotation
- [x] Bounding box mode
- [x] Annotation review system
- [x] Polygon and bbox correction tools
- [x] Undo / Redo
- [x] COCO JSON export
- [x] SQLite feedback database
- [x] HuggingFace auto-backup
- [ ] YOLO `.txt` format export
- [ ] Batch image processing
- [ ] Multi-image dataset accumulation
- [ ] Active learning from user feedback
- [ ] Fine-tuning pipeline on collected data

---

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## License

MIT License — free to use, modify, and distribute for personal and commercial projects.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Meta Segment Anything Model](https://github.com/facebookresearch/segment-anything)
- [FastAPI](https://fastapi.tiangolo.com)

---

## Developers

| Developer | Role | GitHub |
|---|---|---|
| **Mohammed Abdullah** | Backend · AI Pipeline · Deployment | [@0k1nx0](https://github.com/0k1nx0) |
| **Karan Goyal** | Frontend · UI/UX | [@karangoyal09](https://github.com/karangoyal09) |

---

<div align="center">

If this project helped you, please give it a star on GitHub!

Built with YOLOv8 · SAM · FastAPI · Docker

</div>
