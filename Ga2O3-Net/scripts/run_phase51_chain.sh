#!/bin/bash
# Phase 51 chain — Frozen Multi-Expert Fusion via Self-Attention.
#
# Sequential steps:
#   1. WAIT for Phase 50 V18 to finish (gate on logs/phase50_chain/markers/v18.done)
#   2. Pretrain encoder_witman (single-task on V_O f.e., ~5 min)
#   3. Pretrain encoder_jarvis (single-task on TBmBJ + dielectric, ~30 min)
#   4. Smoke test V51 model construction + forward pass on CPU
#   5. Run V51 fine-tune (5 seeds × 10 folds, GPU 0)
#   6. Auto-eval (eval_phase42_vs_baselines + physics_diag_table)
#
# Each step writes a marker; resumable.
#
# Run as:
#   nohup bash scripts/run_phase51_chain.sh > logs/phase51_chain_master.log 2>&1 &

set -e
cd /home/lawrence/Physics/Ga2O3-Net

LOG_DIR="logs/phase51_chain"
MARKER_DIR="logs/phase51_chain/markers"
mkdir -p "$LOG_DIR" "$MARKER_DIR"

CONDA_RUN="conda run -n ga2o3 python -u"

echo "===================================================================="
echo " Phase 51 chain start: $(date)"
echo "===================================================================="

# ---- Step 1: wait for V18 done ----
V18_MARKER="logs/phase50_chain/markers/v18.done"
echo "[$(date)] Step 1/6: Waiting for V18 (Phase 50 chain step 4) to finish..."
echo "  Checking marker: $V18_MARKER"
WAIT_LIMIT=$((12 * 3600))   # 12h hard limit
WAIT_START=$(date +%s)
while [ ! -f "$V18_MARKER" ]; do
    sleep 60
    NOW=$(date +%s)
    if [ $((NOW - WAIT_START)) -gt $WAIT_LIMIT ]; then
        echo "[$(date)] V18 wait timeout (>12h). Aborting."
        exit 1
    fi
done
echo "[$(date)] V18 marker present. Phase 50 chain finished — proceeding."

# ---- Step 2: Pretrain encoder_witman ----
if [ ! -f "$MARKER_DIR/witman_pretrain.done" ]; then
    echo "[$(date)] Step 2/6: Pretraining encoder_witman..."
    CUDA_VISIBLE_DEVICES=0 $CONDA_RUN scripts/02c_pretrain_witman.py \
        --gpu 0 --epochs 200 --batch-size 64 --lr 5.0e-4 \
        --warm-start checkpoints/pretrained_encoder_v2.pt \
        --out checkpoints/pretrained_encoder_witman.pt \
        > "$LOG_DIR/witman_pretrain.log" 2>&1
    touch "$MARKER_DIR/witman_pretrain.done"
    echo "[$(date)] Step 2/6 witman_pretrain: DONE"
fi

# ---- Step 3: Pretrain encoder_jarvis ----
if [ ! -f "$MARKER_DIR/jarvis_pretrain.done" ]; then
    echo "[$(date)] Step 3/6: Pretraining encoder_jarvis..."
    CUDA_VISIBLE_DEVICES=0 $CONDA_RUN scripts/02d_pretrain_jarvis.py \
        --gpu 0 --epochs 60 --batch-size 128 --lr 2.0e-4 \
        --max-atoms 80 \
        --warm-start checkpoints/pretrained_encoder_v2.pt \
        --out checkpoints/pretrained_encoder_jarvis.pt \
        > "$LOG_DIR/jarvis_pretrain.log" 2>&1
    touch "$MARKER_DIR/jarvis_pretrain.done"
    echo "[$(date)] Step 3/6 jarvis_pretrain: DONE"
fi

# ---- Step 4: Smoke test V51 architecture ----
if [ ! -f "$MARKER_DIR/v51_smoke.done" ]; then
    echo "[$(date)] Step 4/6: Smoke testing V51 model..."
    $CONDA_RUN -c "
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
model = build_v51_model(
    encoder_specs=mm['multi_encoder']['encoders'],
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
print(f'V51 forward OK. output shape: {out.shape}')
print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
print(f'Total params:     {sum(p.numel() for p in model.parameters()):,}')
" > "$LOG_DIR/v51_smoke.log" 2>&1
    touch "$MARKER_DIR/v51_smoke.done"
    echo "[$(date)] Step 4/6 v51_smoke: DONE"
fi

# ---- Step 5: Run V51 fine-tune ----
if [ ! -f "$MARKER_DIR/v51_finetune.done" ]; then
    echo "[$(date)] Step 5/6: V51 fine-tune (5 seeds × 10 folds)..."
    CUDA_VISIBLE_DEVICES=0 $CONDA_RUN scripts/04b_finetune_v51.py \
        --multimodal-config config/multimodal_v51.yaml \
        --fusion-config     config/fusion_v51_vc.yaml \
        --gpu 0 --ensemble-seeds 5 --cv-folds 10 --cv-group-by doi \
        --single-target vacancy_concentration \
        --session phase51v51_multiexpert_vc_5seed \
        > "$LOG_DIR/v51_finetune.log" 2>&1
    touch "$MARKER_DIR/v51_finetune.done"
    echo "[$(date)] Step 5/6 v51_finetune: DONE"
fi

# ---- Step 6: Evaluate ----
if [ ! -f "$MARKER_DIR/v51_eval.done" ]; then
    echo "[$(date)] Step 6/6: Evaluating V51..."
    {
        echo "=== eval_phase42 ==="
        $CONDA_RUN scripts/eval_phase42_vs_baselines.py 2>&1 | grep -E "Phase 43 V5|Phase 50 V18|Phase 51|^Mg|^Sn|^Zn|R²_Pl|^Element"
        echo ""
        echo "=== physics_diag_table V51 (scope=all) ==="
        $CONDA_RUN scripts/eval_physics_diag_table.py \
            results/phase51v51_multiexpert_vc_5seed \
            data/raw/experimental/ga2o3_exp_aug3x_vcinterp.csv \
            --target vacancy_concentration --scope all 2>&1 | tail -50
    } > "$LOG_DIR/v51_eval.log" 2>&1
    touch "$MARKER_DIR/v51_eval.done"
    echo "[$(date)] Step 6/6 v51_eval: DONE"
fi

echo "===================================================================="
echo " Phase 51 chain complete: $(date)"
echo "===================================================================="
