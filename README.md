

# RoomReID: Smart Passenger Tracking System

**RoomReID** is an automated end-to-end Person Re-Identification (Re-ID) system designed for tracking and counting individuals across non-overlapping camera views. It utilizes a microservices architecture to process video streams in real-time, handling tasks such as person detection, tracking, and identity matching between Entry and Exit points.

## 🏗 System Architecture

The system is built on a modular microservices architecture orchestrated via Docker Compose:

1. **Perception Engine (DeepStream 7.1):**
* **Core App:** `deepstream-fewshot-learning-app` (C/C++).
* **Detection:** PeopleNet Transformer (Deformable DETR).
* **Tracking:** NvDCF (Discriminative Correlation Filter) with multi-object tracking.
* **Feature Extraction:** ReIdentificationNet (Swin-Tiny) for generating appearance embeddings.
* **Output:** Pushes JSON metadata and feature vectors to Kafka.


2. **Messaging Layer (Apache Kafka):**
* Decouples the high-speed video processing from the database logic.
* Topic `mdx-raw` acts as the buffer for incoming tracklet data.


3. **Data Fusion & Matching (Python):**
* **Consumer:** `kafka-consumer` service reads tracklets from Kafka.
* **Logic:** Executes "Single-Shot" frame selection based on ROI crossing and performs vector similarity matching (Entry Gallery vs. Exit Query).


4. **Vector Database (Milvus):**
* Stores 1024-dim feature vectors.
* Provides high-speed similarity search (HNSW index) for Re-ID.
* Includes **Attu** (Web UI) for database visualization.



## ⚙️ Prerequisites

* **OS:** Linux (Ubuntu 22.04 recommended).
* **Hardware:** NVIDIA GPU (Ampere/Turing architecture recommended).
* **Software:**
  * Docker Engine
  * **NVIDIA Container Toolkit** (Required for GPU access inside containers).
  * NVIDIA Drivers & CUDA Toolkit.



## 🚀 Installation & Deployment

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd roomreid

```

### 2. Model Preparation 

The repository config is set to use Transformer-based models which are **not included** in the repo. You must download them from NVIDIA NGC and place them in the correct directory for the build script to pick them up.

**Directory:** `deepstream/deepstream-fewshot-learning-app/models/`

| Model Type | File Name Required | Source |
| --- | --- | --- |
| **Detection** | `resnet50_peoplenet_transformer_op17.onnx` | [PeopleNet Transformer (NGC)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet_transformer) |
| **Re-ID** | `reid_model_latest.onnx` | [ReIdentificationNet (NGC)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/reidentificationnet) |

**Steps:**

1. Download the `.onnx` files from the links above.
2. Rename them exactly as shown in the table.
3. Ensure `labels.txt` and `detector_labels.txt` are also present in the models folder.

### 3. Build and Start Services

Use Docker Compose to build the custom images and start the stack.

```bash
# Build images and start containers in detached mode
docker compose up -d --build

```

### 4. Verify Deployment

Check the status of the containers:

```bash
docker compose ps

```

You should see the following active services:

* `mdx-deepstream`: Video analytics pipeline.
* `kafka-consumer`: Matching logic.
* `mdx-kafka` / `mdx-kafka-topics`: Data transport.
* `milvus-standalone` / `etcd` / `minio`: Vector DB backend.
* `attu`: Management UI.

## 📂 Configuration

### DeepStream Configuration

Located in: `deepstream/deepstream-fewshot-learning-app/configs/`

* **`mtmc_config.txt`**: The main entry point. Configures inputs (RTSP/File), outputs (Kafka Sink), and plugin linkages.
* **`mtmc_pgie_config_peoplenet_optimized.txt`**: Inference settings for the PeopleNet detector (thresholds, NMS, etc.).
* **`config_tracker_NvDCF_accuracy_peoplenet_optimized.yml`**: Fine-tuned tracker settings, including Shadow Tracking age and Re-ID feature extraction parameters.

### Input Source

To change the video source (File vs. RTSP), edit `mtmc_config.txt`:

```ini
[source-list]
# For RTSP
list=rtsp://127.0.0.1:8554/stream1;rtsp://127.0.0.1:8554/stream2;
# For Files
# list=file:///opt/storage/video1.mp4;file:///opt/storage/video2.mp4

```

## 🛠 Usage & Monitoring

**1. Monitor Logs**
View real-time logs from the DeepStream app to ensure the pipeline is running:

```bash
docker logs -f mdx-deepstream

```

View the Python consumer to see Re-ID matching results:

```bash
docker logs -f kafka-consumer

```

**2. Access Milvus Dashboard**
Open your browser and navigate to `http://localhost:8000` to access **Attu**. This allows you to inspect the feature vectors stored in the database.

## 🔧 Development

* **Startup Script:** The `deepstream/init_scripts/ds-start.sh` script handles model copying (to preserve persistence) and compiling the custom C++ sources if `MODEL_TYPE="transformer"` is set.
* **Custom Parsers:** The `custom_parser/` directory contains C++ code (`nvdsinfer_custombboxparser_tao.cpp`) for parsing the specific output format of the PeopleNet Transformer model.

## 📄 License

See the `LICENSE` file in the `deepstream-fewshot-learning-app` directory for details. Based on NVIDIA sample applications.

## Key Technology References:

* SOLIDER Re-ID Framework: W. Chen et al., "Beyond Appearance: A Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks," in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15050-15061, 2023. 

* Multi-Camera Tracking (MTMC): "Multi-Camera Tracking." NVIDIA Metropolis Microservices Documentation. [Online]. Available: https://docs.nvidia.com/mms/text/MDX_Multi_Camera_Tracking_App.html
