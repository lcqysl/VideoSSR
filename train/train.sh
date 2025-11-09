#!/bin/bash
set -x
ENGINE=${1:-vllm}
PROJECT_NAME="VideoSSR"

MODEL_PATH="path/to/Qwen3-VL-8B-Instruct"
EXPERIMENT_NAME="mixall"
SAVE_CHECKPOINT_DIR="path/to/checkpoints"
TRAIN_FILES=(
    "path/to/parquet/channelswap.parquet"
    "path/to/parquet/mirror.parquet"
    "path/to/parquet/zoomout.parquet"
    "path/to/parquet/rotate180.parquet"
    "path/to/parquet/jigsaw6.parquet"
    "path/to/parquet/counting.parquet"
)

train_files_str=$(printf '"%s",' "${TRAIN_FILES[@]}")
train_files_str=${train_files_str%,}
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files=[${train_files_str}]" \
    data.val_files=path/to/parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    data.val_batch_size=128 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.total_epochs=1 >/home/hezefeng/hezefeng/verl/add/log/${EXPERIMENT_NAME} 2>&1
    