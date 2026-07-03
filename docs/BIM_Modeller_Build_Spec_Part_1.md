# Agentic BIM Modeller — Product & Technical Specification Build Document

**Status:** Part 1 of a multi-part document. This part covers Phase 0 (environment/repo setup) and Phase 1 (SAM data pipeline, resumable fine-tuning, quantization) in full technical depth. Phases 2–8 follow in subsequent parts, appended to this same file.

**How to read this document:** Phases are strictly sequential — do not start a phase before the previous one's "Definition of Done" is met. Every phase begins with prerequisites, ends with a concrete completion checklist, and every non-trivial step includes real, working code — not pseudocode.

---

## TABLE OF CONTENTS

- Phase 0 — Environment & Repository Setup
- Phase 1 — SAM Data Pipeline, Resumable Fine-Tuning, Quantization
- Phase 2 — SAM Evaluation + Deterministic Validator *(next part)*
- Phase 3 — Planning Agent + Constraint Engine *(next part)*
- Phase 4 — IFC/DXF Export + File Storage *(next part)*
- Phase 5 — Orchestration Graph + Repair Agent + RAG KB *(next part)*
- Phase 6 — Parametric Geometry Engine + Async Workers *(next part)*
- Phase 7 — Coding Agent + Script Safety Layer *(next part)*
- Phase 8 — Plugin Bridge *(next part)*

---

# PHASE 0 — ENVIRONMENT & REPOSITORY SETUP

**Prerequisite:** Nothing. This is the true starting point — no repo, no environment.

### 0.1 Repository Layout

Create this exact structure. Every later phase writes into a predetermined folder — decide it once now so nothing is improvised later.

```
bim-modeller/
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── .gitignore
├── sam/                    # Phase 1–2: data pipeline, training, evaluation, validator
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── serving/
├── agents/                 # Phase 3, 5, 7: Planning Agent, Repair Agent, Coding Agent
├── orchestration/          # Phase 5: graph, state schema
├── constraints/            # Phase 3: Constraint Engine + IBC rule set
├── geometry/                # Phase 6: parametric + solid geometry engine
├── export/                  # Phase 4: IFC/DXF authoring
├── api/                      # Phase 0+: FastAPI gateway
├── plugin_bridge/            # Phase 8: Revit/Rhino bridge (empty until Phase 8)
├── infra/                    # Docker, tracing/metrics config
└── docs/
```

Run:
```bash
mkdir -p bim-modeller/{sam/{data,training,evaluation,serving},agents,orchestration,constraints,geometry,export,api,plugin_bridge,infra,docs}
cd bim-modeller && git init
```

### 0.2 Python Environment

Use `uv` (faster, more reliable dependency resolution than plain pip/venv — a reasonable "mature off-the-shelf" substitution for basic tooling).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init --python 3.11
```

`pyproject.toml` — core dependencies for Phase 0–1 only (later phases add their own, introduced when needed):

```toml
[project]
name = "bim-modeller"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "transformers>=4.44",
    "peft>=0.12",
    "bitsandbytes>=0.43",
    "trl>=0.10",
    "datasets>=2.20",
    "accelerate>=0.33",
    "shapely>=2.0",
    "pydantic>=2.8",
    "fastapi>=0.112",
    "uvicorn>=0.30",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5"]
```

```bash
uv sync
```

### 0.3 Supporting Services (self-hosted, per Section 2.3)

`docker-compose.yml` — brings up everything Phase 4–7 will need later. Start it now so the containers exist from day one, even though most sit unused until their phase arrives.

```yaml
version: "3.9"
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: bim
      POSTGRES_PASSWORD: bim_dev_only
      POSTGRES_DB: bim_modeller
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]

  redis:
    image: redis:7
    ports: ["6379:6379"]

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: bim
      MINIO_ROOT_PASSWORD: bim_dev_only
    ports: ["9000:9000", "9001:9001"]
    volumes: ["miniodata:/data"]

  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]

volumes:
  pgdata:
  miniodata:
```

```bash
docker compose up -d
```

### 0.4 Environment Variables

`.env.example` (copy to `.env`, fill in secrets — never commit `.env`):

```
DATABASE_URL=postgresql://bim:bim_dev_only@localhost:5432/bim_modeller
REDIS_URL=redis://localhost:6379/0
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=bim
MINIO_SECRET_KEY=bim_dev_only
QDRANT_URL=http://localhost:6333
HF_TOKEN=                       # needed to download gated base models (e.g. Llama 3.1)
```

`.gitignore`:
```
.env
__pycache__/
*.pyc
.venv/
sam/data/processed/
sam/training/checkpoints/
*.gguf
*.arrow
```

### 0.5 Definition of Done — Phase 0

- [ ] Repo structure created, `git init` done, first commit made
- [ ] `uv sync` completes cleanly
- [ ] `docker compose up -d` brings up postgres, redis, minio, qdrant with no errors
- [ ] `.env` populated with a valid `HF_TOKEN` (required for Phase 1 — Llama 3.1 is gated on Hugging Face; request access before Phase 1 starts, approval can take time)

---

# PHASE 1 — SAM DATA PIPELINE, RESUMABLE FINE-TUNING, QUANTIZATION

**Prerequisite:** Phase 0 complete. **Do not proceed to Phase 2 until this phase's Definition of Done is met** — per the roadmap, nothing downstream should be built against an unproven SAM.

## 1.1 Acquire the Raw Dataset

Source your primary dataset (real residential floor plans with room polygons, type labels, adjacency graph, and site boundary — per Bible Section 3.4). Place the raw files under:

```
sam/data/raw/
```

The preprocessing pipeline below is written against the schema described in the Bible — adapt field names in `RawPlan` (Section 1.2 below) to whatever your actual source files use; the logic does not change.

## 1.2 Preprocessing Pipeline

This directly implements every finding in Bible Section 3.5. Each fix is its own function so you can unit-test them independently.

`sam/data/preprocess.py`:

```python
"""
Preprocessing pipeline for raw floor plan data -> clean training pairs.
Implements Bible Section 3.5 findings, in order:
  1. Coordinate scale correction (pixels -> meters)
  2. Background/void field exclusion
  3. Fragmented room polygon merging
  4. Duplicate overlapping room detection/removal
  5. Adjacency graph pass-through (never re-derived)
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# Fields in the raw dataset that represent unbuilt/background space,
# NOT architectural rooms. Confirm against your actual source's labels.
BACKGROUND_LABELS = {"background", "void", "unlabeled", "exterior"}

# Sanity bounds (sqm) used to catch scale-conversion errors early.
ROOM_AREA_SANITY = {
    "bedroom": (6.0, 50.0),
    "bathroom": (2.0, 15.0),
    "kitchen": (4.0, 40.0),
    "living": (10.0, 80.0),
    "default": (1.5, 100.0),
}


@dataclass
class RawRoom:
    room_type: str
    polygons_px: list[list[tuple[float, float]]]  # may be >1 fragment


@dataclass
class RawPlan:
    plan_id: str
    site_boundary_px: list[tuple[float, float]]
    site_area_sqm: float          # ground-truth real-world site area, from source metadata
    rooms: list[RawRoom]
    adjacency_pairs: list[tuple[str, str]]  # pre-computed, from source — do not re-derive


@dataclass
class CleanRoom:
    name: str
    room_type: str
    zone: str
    area: float
    polygon: list[tuple[float, float]]  # meters, closed ring


@dataclass
class CleanPlan:
    plan_id: str
    site_width: float
    site_length: float
    rooms: list[CleanRoom]
    adjacency: list[tuple[str, str]]
    total_area: float


ZONE_MAP = {
    "bedroom": "private", "bathroom": "private",
    "living": "social", "kitchen": "social", "dining": "social",
    "entry": "circulation", "corridor": "circulation",
}


def compute_scale_factor(site_boundary_px: list[tuple[float, float]], site_area_sqm: float) -> float:
    """Bible 3.5 #1: coordinates are pixel-space by default.
    scale = sqrt(real_world_area / pixel_space_area_of_site_boundary)
    """
    site_poly_px = Polygon(site_boundary_px)
    pixel_area = site_poly_px.area
    if pixel_area <= 0:
        raise ValueError("Degenerate site boundary polygon")
    return math.sqrt(site_area_sqm / pixel_area)


def scale_ring(ring_px: list[tuple[float, float]], scale: float) -> list[tuple[float, float]]:
    return [(x * scale, y * scale) for x, y in ring_px]


def merge_fragments(polygons_px: list[list[tuple[float, float]]], scale: float) -> Polygon | None:
    """Bible 3.5 #3: merge disconnected polygon fragments of the same room.
    Union first; fall back to convex hull if union is still a disconnected MultiPolygon.
    """
    scaled = [Polygon(scale_ring(p, scale)) for p in polygons_px if len(p) >= 3]
    scaled = [p for p in scaled if p.is_valid and p.area > 0]
    if not scaled:
        return None
    merged = unary_union(scaled)
    if isinstance(merged, MultiPolygon):
        # still disconnected after union -> fall back to convex hull around all fragments
        all_points = [pt for poly in scaled for pt in poly.exterior.coords]
        merged = Polygon(all_points).convex_hull
    return merged


def is_duplicate(room_a: CleanRoom, room_b: CleanRoom) -> bool:
    """Bible 3.5 #4: identical coordinate lists = data defect, discard the example."""
    return room_a.polygon == room_b.polygon


def sanity_check_area(room_type: str, area: float) -> bool:
    lo, hi = ROOM_AREA_SANITY.get(room_type, ROOM_AREA_SANITY["default"])
    return lo <= area <= hi


def preprocess_plan(raw: RawPlan) -> CleanPlan | None:
    """Returns None if the plan fails validation and should be discarded outright."""
    scale = compute_scale_factor(raw.site_boundary_px, raw.site_area_sqm)

    site_poly = Polygon(scale_ring(raw.site_boundary_px, scale))
    minx, miny, maxx, maxy = site_poly.bounds
    site_width, site_length = maxx - minx, maxy - miny

    clean_rooms: list[CleanRoom] = []
    type_counts: dict[str, int] = {}

    for raw_room in raw.rooms:
        # Bible 3.5 #2: exclude background/void fields — they are not rooms.
        if raw_room.room_type.lower() in BACKGROUND_LABELS:
            continue

        merged = merge_fragments(raw_room.polygons_px, scale)
        if merged is None or merged.is_empty:
            continue

        area = round(merged.area, 2)
        if not sanity_check_area(raw_room.room_type.lower(), area):
            # Scale conversion or extraction error — do not silently include bad data.
            continue

        idx = type_counts.get(raw_room.room_type.lower(), 0)
        type_counts[raw_room.room_type.lower()] = idx + 1
        name = f"{raw_room.room_type.lower()}_{idx}"

        clean_rooms.append(CleanRoom(
            name=name,
            room_type=raw_room.room_type.lower(),
            zone=ZONE_MAP.get(raw_room.room_type.lower(), "other"),
            area=area,
            polygon=[(round(x, 3), round(y, 3)) for x, y in merged.exterior.coords],
        ))

    # Bible 3.5 #4: discard the whole example if any duplicate overlapping rooms exist.
    for i in range(len(clean_rooms)):
        for j in range(i + 1, len(clean_rooms)):
            if is_duplicate(clean_rooms[i], clean_rooms[j]):
                return None

    if not clean_rooms:
        return None

    # Bible 3.5 #5: adjacency graph is pre-computed upstream — pass through, never re-derive.
    adjacency = [pair for pair in raw.adjacency_pairs]

    return CleanPlan(
        plan_id=raw.plan_id,
        site_width=round(site_width, 2),
        site_length=round(site_length, 2),
        rooms=clean_rooms,
        adjacency=adjacency,
        total_area=round(sum(r.area for r in clean_rooms), 2),
    )


def run_pipeline(raw_dir: Path, out_path: Path) -> None:
    clean_plans = []
    discarded = 0
    for raw_file in raw_dir.glob("*.json"):
        raw_data = json.loads(raw_file.read_text())
        raw = RawPlan(**raw_data)  # adapt to your actual raw schema
        clean = preprocess_plan(raw)
        if clean is None:
            discarded += 1
            continue
        clean_plans.append(clean)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for plan in clean_plans:
            f.write(json.dumps(plan.__dict__, default=lambda o: o.__dict__) + "\n")

    print(f"Processed {len(clean_plans)} valid plans, discarded {discarded}.")


if __name__ == "__main__":
    run_pipeline(Path("sam/data/raw"), Path("sam/data/processed/clean_plans.jsonl"))
```

Run it, and **spot-check output manually** before moving on — pull 10 random plans from `clean_plans.jsonl` and visually plot them (e.g. with `shapely` + `matplotlib`) to confirm areas and shapes look like real rooms, not corrupted geometry.

## 1.3 Format Training Pairs

Convert clean plans into the exact input/output schema from Bible Section 3.6, as instruction-tuning chat turns.

`sam/data/format_training_pairs.py`:

```python
import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a spatial layout generator for residential floor plans. "
    "Given a site and a brief, output ONLY a JSON object with keys: "
    "rooms, adjacency, circulation, total_area. No prose, no markdown fences."
)

def plan_to_training_pair(plan: dict) -> dict:
    site = {"width": plan["site_width"], "length": plan["site_length"]}
    brief = build_brief_text(plan)  # see below

    user_content = json.dumps({
        "site": site,
        "brief": brief,
        "constraints": {"front_setback": 3.0, "coverage_ratio": 0.65},  # pull real values if available
    })

    assistant_content = json.dumps({
        "rooms": [
            {
                "name": r["name"], "type": r["room_type"], "zone": r["zone"],
                "area": r["area"], "polygon": r["polygon"], "floor": 1,
            }
            for r in plan["rooms"]
        ],
        "adjacency": plan["adjacency"],
        "circulation": infer_circulation_text(plan),
        "total_area": plan["total_area"],
    })

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def build_brief_text(plan: dict) -> str:
    counts: dict[str, int] = {}
    for r in plan["rooms"]:
        counts[r["room_type"]] = counts.get(r["room_type"], 0) + 1
    parts = [f"{n} {t}" + ("s" if n > 1 else "") for t, n in counts.items()]
    return f"Residential unit on a {plan['site_width']}m x {plan['site_length']}m site with " + ", ".join(parts) + "."


def infer_circulation_text(plan: dict) -> str:
    # Minimal placeholder — refine once real circulation metadata is available in the source dataset.
    zones_order = sorted(plan["rooms"], key=lambda r: r["zone"])
    return " -> ".join(dict.fromkeys(r["zone"] for r in zones_order))


def run(in_path: Path, out_path: Path) -> None:
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            plan = json.loads(line)
            pair = plan_to_training_pair(plan)
            fout.write(json.dumps(pair) + "\n")


if __name__ == "__main__":
    run(Path("sam/data/processed/clean_plans.jsonl"), Path("sam/data/processed/training_pairs.jsonl"))
```

## 1.4 Tokenize Once, Cache Permanently

This is the fix for wasted Kaggle session time re-tokenizing 23k examples every session.

`sam/training/tokenize_and_cache.py`:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # swap per Section 1.6 base-model comparison
MAX_LEN = 2048

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files="sam/data/processed/training_pairs.jsonl", split="train")

    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        tokens = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=4)
    ds = ds.train_test_split(test_size=0.05, seed=42)
    ds.save_to_disk("sam/data/processed/tokenized_dataset")
    print(ds)

if __name__ == "__main__":
    main()
```

Persist `sam/data/processed/tokenized_dataset/` as its own Kaggle Dataset immediately after this runs once — every future session loads it in seconds instead of re-running this step.

## 1.5 The Resumable QLoRA Training Harness

This directly implements the audit's fix for the checkpoint/resume bug: full trainer-managed state, a time-budget stop before Kaggle's kill, and adapter-only checkpoints small enough to commit every session.

`sam/training/train_sam.py`:

```python
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainerCallback, TrainerControl, TrainerState, TrainingArguments,
)
from trl import SFTTrainer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CHECKPOINT_DIR = "sam/training/checkpoints"
SESSION_TIME_BUDGET_SECONDS = 7 * 3600 - 15 * 60  # stop 15 min before Kaggle's 7-hour limit


class TimeBudgetCallback(TrainerCallback):
    """Forces a clean checkpoint + stop before the environment kills the process."""

    def __init__(self, budget_seconds: int):
        self.budget_seconds = budget_seconds
        self.start_time = time.time()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.budget_seconds:
            control.should_save = True
            control.should_training_stop = True
        return control


def load_base_model_4bit():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
    )
    return model


def get_lora_config() -> LoraConfig:
    # Exactly matches Bible Section 3.3: r=16, alpha=16, all projection layers.
    return LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def find_latest_checkpoint() -> str | None:
    ckpt_root = Path(CHECKPOINT_DIR)
    if not ckpt_root.exists():
        return None
    checkpoints = sorted(ckpt_root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk("sam/data/processed/tokenized_dataset")

    resume_path = find_latest_checkpoint()

    base_model = load_base_model_4bit()

    if resume_path:
        # Resume: load adapter as trainable, not frozen — the #1 PEFT+Trainer resume gotcha.
        print(f"Resuming from {resume_path}")
        model = PeftModel.from_pretrained(base_model, resume_path, is_trainable=True)
    else:
        print("Starting fresh run")
        model = get_peft_model(base_model, get_lora_config())

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,          # bound disk usage — old checkpoints auto-pruned
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        bf16=True,
        report_to=[],                 # wire to LLM tracing tool once Phase 5 sets it up
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        callbacks=[TimeBudgetCallback(SESSION_TIME_BUDGET_SECONDS)],
    )

    trainer.train(resume_from_checkpoint=resume_path)

    # Final adapter-only save for this session (small — tens of MB, fast to commit).
    trainer.save_model(f"{CHECKPOINT_DIR}/latest_adapter")


if __name__ == "__main__":
    main()
```

### 1.5.1 Kaggle Session Commit/Resume Cycle

At the **end of every Kaggle session** (last notebook cell):

```bash
kaggle datasets version -p sam/training/checkpoints -m "session checkpoint update" -d
```

At the **start of every new session**, add that dataset as a notebook input, then:

```bash
cp -r /kaggle/input/<your-checkpoint-dataset-slug>/* sam/training/checkpoints/
python sam/training/train_sam.py
```

`find_latest_checkpoint()` in the script above automatically detects and resumes from whatever was copied in — no manual path editing required.

## 1.6 Base Model Comparison (per prior discussion)

Before committing to a full-scale run, repeat Sections 1.4–1.5 on a ~1,000-example subset for each candidate, changing only `BASE_MODEL`:

- `meta-llama/Llama-3.1-8B-Instruct` (spec default)
- `HuggingFaceTB/SmolLM3-3B`
- `Qwen/Qwen3-4B`

Score each against the Phase 2 validator's pass rate (built next) on a held-out set. Pick the winner before running the full 23k-example fine-tune.

## 1.7 Post-Training: Merge, then Quantize to GGUF

Quantization happens only after fine-tuning is complete and finished (Bible Section 3.3) — the QLoRA 4-bit base used *during training* is a separate, training-only efficiency technique and is not the deployment artifact.

`sam/training/merge_and_export.py`:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "sam/training/checkpoints/latest_adapter"
MERGED_OUT = "sam/training/merged_fp16"

def main():
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    merged = model.merge_and_unload()  # bakes adapter into full-precision weights
    merged.save_pretrained(MERGED_OUT)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(MERGED_OUT)

if __name__ == "__main__":
    main()
```

Convert and quantize using `llama.cpp` (mature off-the-shelf tool per Section 11.1 — do not hand-roll a quantizer):

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release

python convert_hf_to_gguf.py ../sam/training/merged_fp16 --outfile ../sam/serving/sam.f16.gguf

./build/bin/llama-quantize ../sam/serving/sam.f16.gguf ../sam/serving/sam.Q4_K_M.gguf Q4_K_M
```

### 1.7.1 Serve via Ollama

`sam/serving/Modelfile`:
```
FROM ./sam.Q4_K_M.gguf
PARAMETER temperature 0.3
PARAMETER num_ctx 2048
SYSTEM "You are a spatial layout generator for residential floor plans. Output ONLY valid JSON with keys: rooms, adjacency, circulation, total_area."
```

```bash
ollama create sam-v1 -f sam/serving/Modelfile
ollama run sam-v1
```

## 1.8 Definition of Done — Phase 1

- [ ] `clean_plans.jsonl` produced, manually spot-checked (10+ plans visually plotted, areas look real)
- [ ] `training_pairs.jsonl` produced in the exact Section 3.6 schema
- [ ] Tokenized dataset cached and persisted as its own Kaggle Dataset
- [ ] Resumable training harness proven: kill mid-run, restart, confirm `global_step` continues rather than resetting to 0
- [ ] Base model comparison run (Section 1.6), a winner selected with a documented reason
- [ ] Full-scale fine-tune completed on the winning base model
- [ ] `sam.Q4_K_M.gguf` produced and runs locally via `ollama run sam-v1`, returning syntactically valid JSON for at least a simple test brief

**Do not proceed to Phase 2 until every box above is checked.**

---

*End of Part 1. Part 2 continues with Phase 2 (SAM Evaluation + Deterministic Validator) and Phase 3 (Planning Agent + Constraint Engine).*
