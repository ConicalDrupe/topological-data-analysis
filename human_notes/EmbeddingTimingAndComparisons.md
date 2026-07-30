What actually got measured
RTX 3090 (24GB VRAM), SigLIP so400m-patch14-384, bf16, batch size 16:

Split	Images	Wall time (incl. ~3s model load)	Throughput
test	115	19.5s	~7 img/s
train	460	~65s (extrapolated from the same run rate)	~7 img/s
Full cohort (575 images total) is under 90 seconds end-to-end — this dataset is tiny by GPU standards, so wall-clock time here is completely dominated by model load, not compute. Your 3090 is massive overkill for these model sizes; nothing here will be VRAM- or throughput-constrained.

How it'll compare to RAD-DINO and MedGemma
Pulled parameter counts from the public model cards (haven't run these two yet — both are gated):

Backend	Params (used)	Resolution	Notes
SigLIP so400m (just ran)	428M (vision-only; 878M total but text tower is unused/wasted)	384×384	~7 img/s measured above
RAD-DINO	86.6M (DINOv2-base)	518×518	~5x fewer params than SigLIP's vision tower, but ~1.8x the pixels. Net: still expect it faster than SigLIP, maybe 2-3x — call it test split in ~8-12s once you have access. Real number needs an actual run.
MedGemma-4B-it	vision tower is a medically-fine-tuned SigLIP, but encoders.py currently loads the entire 4B-parameter multimodal model via AutoModel/AutoModelForImageTextToText even though only vision_tower gets used	896×896 (→4096 patches pre-compression, pooled to 256 tokens)	This will be the slowest and heaviest by a good margin: ~4.5x more weights to load off disk than SigLIP, and a vision forward pass over ~5.6x more patch tokens than SigLIP's 384² grid before any pooling. Still fits comfortably in 24GB (~8GB just for bf16 weights), just slower per image.