# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                          local_pipeline_manual.py                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

import warnings
from pathlib import Path
from types import SimpleNamespace
from pyannote.audio.core.model import Specifications

import torch
from safetensors.torch import load_file

from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.embedding import XVectorSincNet
from pyannote.audio.pipelines import SpeakerDiarization

# Ignore harmless torchaudio backend warnings
warnings.filterwarnings("ignore", message="torchaudio._backend.list_audio_backends")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                          CONFIGURATION (EDIT ME)                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝
BASE = Path("D:/WIP/convert_safetensor")
SEG_SF = BASE / "segmentation" / "model.safetensors"
EMB_SF = BASE / "embedding" / "model.safetensors"
AUDIO_FILE = "multi_speaker_overlapped.wav"   # relative or absolute path to sample


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                 1) Load Segmentation Model Weights (Safetensors)           ║
# ╚════════════════════════════════════════════════════════════════════════════╝
seg_state = load_file(SEG_SF.as_posix())

# Try to auto-detect number of classes
if "classifier.weight" in seg_state:
    num_classes = int(seg_state["classifier.weight"].shape[0])
else:
    possible = [k for k in seg_state.keys() if "classifier" in k or "output" in k]
    if possible:
        w = seg_state[possible[0]]
        num_classes = int(w.shape[0])
    else:
        num_classes = 2   # default: binary speech/non-speech

print(f"Auto-detected segmentation num_classes = {num_classes}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                  2) Build & Load Segmentation Model (PyanNet)              ║
# ╚════════════════════════════════════════════════════════════════════════════╝
segmentation_model = PyanNet(
    sample_rate=16000,
    num_channels=1,
    sincnet={"stride": 10},
    lstm={"hidden_size": 128, "num_layers": 4, "bidirectional": True, "monolithic": True},
    linear={"hidden_size": 128, "num_layers": 2},
)

# Load weights (ignore training-only heads)
missing, unexpected = segmentation_model.load_state_dict(seg_state, strict=False)
segmentation_model.eval()
print("Segmentation model loaded.")
print("  missing:", missing)
print("  unexpected:", unexpected)

# Add pyannote-required model specifications
segmentation_model.specifications = Specifications(
    problem="segmentation",
    resolution=0.01,   # 10 ms frames
    duration=10.0,
    classes=num_classes,
)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                     3) Build & Load Embedding Model                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
embedding_model = XVectorSincNet()
emb_state = load_file(EMB_SF.as_posix())

# Clean out training-only parameters
clean_emb = {
    k: v for k, v in emb_state.items()
    if not k.startswith("loss_func.") and not k.startswith("metric_loss")
}

missing_e, unexpected_e = embedding_model.load_state_dict(clean_emb, strict=False)
embedding_model.eval()
print("Embedding model loaded.")
print("  missing:", missing_e)
print("  unexpected:", unexpected_e)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                     4) Build & Run Speaker Diarization                     ║
# ╚════════════════════════════════════════════════════════════════════════════╝
pipeline = SpeakerDiarization(
    segmentation=segmentation_model,
    embedding=embedding_model,
    embedding_batch_size=32,
    segmentation_batch_size=32,
)

# Optional tuning
pipeline._params["clustering"]["method"] = "centroid"
pipeline._params["clustering"]["threshold"] = 0.70

print("Running diarization — using ONLY local models (no network).")
diarization = pipeline(AUDIO_FILE)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           5) Save & Display Output                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝
out_rttm = Path(AUDIO_FILE).stem + "_local.rttm"
with open(out_rttm, "w") as f:
    diarization.write_rttm(f)

print("Wrote", out_rttm)
print("\nSegments:")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.2f} — {turn.end:.2f}   {speaker}")
