# SARM training
# export POLICY=sarm
export DATASET_NAME=normal_0

# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --policy.annotation_mode=single_stage --policy.image_key=observation.images.front --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --batch_size=32 --steps=5000 --wandb.enable=true --wandb.project=sarm --wandb.disable_artifact=true

# Compute RABC weights
# uv run lerobot/src/lerobot/policies/sarm/compute_rabc_weights.py --dataset-repo-id local/${DATASET_NAME} --dataset-root datasets/${DATASET_NAME} --reward-model-path outputs/train/sarm_normal_0/checkpoints/005000/pretrained_model --head-mode sparse --num-visualizations 5

# export POLICY=smolvla
# uv run lerobot-train --dataset.repo_id=local/${DATASET_NAME} --dataset.root=datasets/${DATASET_NAME} --policy.type=$POLICY --output_dir=outputs/train/${POLICY}_${DATASET_NAME} --job_name=${POLICY}_${DATASET_NAME} --policy.device=cuda --policy.push_to_hub=false --wandb.enable=true --wandb.disable_artifact=true --batch_size=32 --steps=100000 --use_rabc=true --rabc_head_mode=sparse --rabc_kappa=0.01 --dataset.video_backend=pyav

export POLICY=xvla
uv run lerobot-train \
    --dataset.repo_id=local/${DATASET_NAME} \
    --dataset.root=datasets/${DATASET_NAME} \
    --output_dir=./outputs/train/${POLICY}_${DATASET_NAME} \
    --job_name=${POLICY}_${DATASET_NAME} \
    --policy.path="lerobot/xvla-base" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --policy.dtype=bfloat16 \
    --policy.action_mode=auto \
    --batch_size=8 \
    --steps=300000 \
    --policy.freeze_vision_encoder=false \
    --policy.freeze_language_encoder=false \
    --policy.train_policy_transformer=true \
    --policy.train_soft_prompts=true \
    --dataset.video_backend=pyav \
    --rename_map='{"observation.images.front": "observation.images.image", "observation.images.side": "observation.images.image2"}'