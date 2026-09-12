#!/bin/bash
# Phase 51 chain on GPU 1 — runs in PARALLEL with V18 on GPU 0.
#
# No conflict with V18:
#   - Different GPU (1 vs 0)
#   - Different session output dir (phase51v51_* vs phase50v18_*)
#   - Different checkpoint files (encoder_witman/jarvis NEW, encoder_v2 read-only)
#   - Different graph cache dirs (witman_graphs/, jarvis_graphs/ NEW)
#
# Sequential steps:
#   1. Pretrain encoder_witman on Witman/Goyal V_O f.e. (~5 min)
#   2. Pretrain encoder_jarvis on JARVIS-DFT (~30 min)
#   3. Smoke test V51 model construction + forward pass on GPU 1
#   4. Run V51 fine-tune (5 seeds × 10 folds, GroupKFold by DOI, ~3-5h)
#   5. Auto-eval (eval_phase42 + physics_diag)

set -e
cd /home/lawrence/Physics/Ga2O3-Net

LOG_DIR="logs/phase51_chain"
MARKER_DIR="logs/phase51_chain/markers"
mkdir -p "$LOG_DIR" "$MARKER_DIR"

CONDA_RUN="conda run -n ga2o3 python -u"
GPU=1

echo "===================================================================="
echo " Phase 51 chain (GPU $GPU) start: $(date)"
echo "===================================================================="

# ---- Step 1: Pretrain encoder_witman ----
if [ ! -f "$MARKER_DIR/witman_pretrain.done" ]; then
    echo "[$(date)] Step 1/5: Pretraining encoder_witman on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU $CONDA_RUN scripts/02c_pretrain_witman.py \
        --gpu 0 --epochs 200 --batch-size 64 --lr 5.0e-4 \
        --warm-start checkpoints/pretrained_encoder_v2.pt \
        --out checkpoints/pretrained_encoder_witman.pt \
        > "$LOG_DIR/witman_pretrain.log" 2>&1
    touch "$MARKER_DIR/witman_pretrain.done"
    echo "[$(date)] Step 1/5 witman_pretrain: DONE"
fi

# ---- Step 2: Pretrain encoder_jarvis ----
if [ ! -f "$MARKER_DIR/jarvis_pretrain.done" ]; then
    echo "[$(date)] Step 2/5: Pretraining encoder_jarvis on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU $CONDA_RUN scripts/02d_pretrain_jarvis.py \
        --gpu 0 --epochs 60 --batch-size 128 --lr 2.0e-4 \
        --max-atoms 80 \
        --warm-start checkpoints/pretrained_encoder_v2.pt \
        --out checkpoints/pretrained_encoder_jarvis.pt \
        > "$LOG_DIR/jarvis_pretrain.log" 2>&1
    touch "$MARKER_DIR/jarvis_pretrain.done"
    echo "[$(date)] Step 2/5 jarvis_pretrain: DONE"
fi

# ---- Step 3: Smoke test V51 architecture on GPU 1 ----
if [ ! -f "$MARKER_DIR/v51_smoke_gpu1.done" ]; then
    echo "[$(date)] Step 3/5: Smoke testing V51 model on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU $CONDA_RUN -c "
import sys; sys.path.insert(0, '/home/lawrence/Physics/Ga2O3-Net')
import torch
from src.data.experimental_dataset import Ga2O3ExpDataset, collate_fn
from src.models.ga2o3_net_v51 import build_v51_model
from torch.utils.data import DataLoader
import yaml
mm = yaml.safe_load(open('config/multimodal_v51.yaml'))
fu = yaml.safe_load(open('config/fusion_v51_vc.yaml'))
ds = Ga2O3ExpDataset(
    csv_path=fu['paths']['experimental_csv'],
    structures_dir=fu['paths']['structures_dir'],
    target_cols=fu['targets'],
    augment=False, prefer_ordered=False,
    exclude_elements=fu.get('data', {}).get('exclude_elements', None),
)
specs = mm['multi_encoder']['encoders']
import os
proj = '/home/lawrence/Physics/Ga2O3-Net'
specs = [{**s, 'ckpt_path': os.path.join(proj, s['ckpt_path']) if not os.path.isabs(s['ckpt_path']) else s['ckpt_path']} for s in specs]
model = build_v51_model(
    encoder_specs=specs,
    target_cols=fu['targets'],
    fusion_dim=mm['multi_encoder']['attn_dim'],
    fusion_pool=mm['multi_encoder']['pool'],
    n_attn_heads=mm['multi_encoder']['n_heads'],
    n_attn_layers=mm['multi_encoder']['n_layers'],
    fusion_mode=mm['fusion']['mode'],
    composition_kwargs={'hidden_dim':128,'out_dim':64,'dropout':0.1},
    process_kwargs={'in_dim':18,'hidden_dim':64,'out_dim':32,'dropout':0.2,
                    'continuous_dim':12,'method_embed_dim':3,'n_methods':6,
                    'substrate_embed_dim':3,'n_substrates':7},
    head_hidden_dims=[64,32], head_dropout=0.30, head_type='brouwer',
    brouwer_use_method_offset=True, brouwer_n_methods=6,
).to('cuda:0')
loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
batch = next(iter(loader))
batch = {k: (v.to('cuda:0') if torch.is_tensor(v) else v) for k,v in batch.items()}
batch['graph'] = batch['graph'].to('cuda:0')
out = model(batch['graph'], batch['dopant_spec'], batch['process'])
print(f'V51 forward OK on GPU $GPU. output shape: {out.shape}')
print(f'Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
print(f'Total:     {sum(p.numel() for p in model.parameters()):,}')
" > "$LOG_DIR/v51_smoke_gpu1.log" 2>&1
    touch "$MARKER_DIR/v51_smoke_gpu1.done"
    echo "[$(date)] Step 3/5 v51_smoke: DONE"
fi

# ---- Step 4: Run V51 fine-tune on GPU 1 ----
if [ ! -f "$MARKER_DIR/v51_finetune.done" ]; then
    echo "[$(date)] Step 4/5: V51 fine-tune (5 seeds × 10 folds) on GPU $GPU..."
    CUDA_VISIBLE_DEVICES=$GPU $CONDA_RUN scripts/04b_finetune_v51.py \
        --multimodal-config config/multimodal_v51.yaml \
        --fusion-config     config/fusion_v51_vc.yaml \
        --gpu 0 --ensemble-seeds 5 --cv-folds 10 --cv-group-by doi \
        --single-target vacancy_concentration \
        --session phase51v51_multiexpert_vc_5seed \
        > "$LOG_DIR/v51_finetune.log" 2>&1
    touch "$MARKER_DIR/v51_finetune.done"
    echo "[$(date)] Step 4/5 v51_finetune: DONE"
fi

# ---- Step 5: Evaluate ----
if [ ! -f "$MARKER_DIR/v51_eval.done" ]; then
    echo "[$(date)] Step 5/5: Evaluating V51..."
    {
        echo "=== eval_phase42_vs_baselines (V5 vs V51) ==="
        $CONDA_RUN scripts/eval_phase42_vs_baselines.py 2>&1 | grep -E "Phase 43 V5|Phase 50 V18|Phase 51|^Mg|^Sn|^Zn|R²_Pl|^Element"
        echo ""
        echo "=== physics_diag_table V51 (scope=all) ==="
        $CONDA_RUN scripts/eval_physics_diag_table.py \
            results/phase51v51_multiexpert_vc_5seed \
            data/raw/experimental/ga2o3_exp_aug3x_vcinterp.csv \
            --target vacancy_concentration --scope all 2>&1 | tail -50
    } > "$LOG_DIR/v51_eval.log" 2>&1
    touch "$MARKER_DIR/v51_eval.done"
    echo "[$(date)] Step 5/5 v51_eval: DONE"
fi

echo "===================================================================="
echo " Phase 51 chain (GPU $GPU) complete: $(date)"
echo "===================================================================="
