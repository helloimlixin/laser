#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Projects/laser

stamp="$(date +%Y%m%d_%s%3N)"
root="/workspace/Projects/laser/outputs/imagenet-rqvae-4gpu-a2048k2-adv10-fixed-${stamp}"
clean_ckpt="/workspace/Projects/laser/outputs/imagenet-rqvae-4gpu-a2048k2-rqarch-20260716_1784224786672/checkpoints/run_20260716_180043/laser/last.ckpt"

exec .venv/bin/python train.py stage1 \
  model=laser_image_nonpatch_d5 data=imagenet \
  output_dir="$root" seed=42 \
  data.data_dir=/workspace/Projects/data/imagenet data.image_size=256 \
  data.batch_size=32 data.eval_batch_size=64 data.num_workers=24 \
  train.max_epochs=10 train.max_steps=-1 train.accelerator=gpu train.devices=4 \
  train.num_nodes=1 train.strategy=ddp train.precision=bf16-mixed \
  train.learning_rate=4.0e-5 model.dict_learning_rate=4.0e-5 \
  train.beta=0.5 train.beta2=0.9 train.warmup_steps=5005 train.min_lr_ratio=1.0 \
  train.gradient_clip_val=0.0 train.accumulate_grad_batches=1 \
  train.limit_train_batches=1.0 train.limit_val_batches=512 train.limit_test_batches=0 \
  train.val_check_interval=1.0 train.run_test_after_fit=false train.compute_rfid_after_fit=false \
  model.backbone=ddpm model.num_hiddens=128 model.num_residual_blocks=2 \
  model.num_residual_hiddens=96 model.num_downsamples=5 \
  'model.channel_multipliers=[1,1,2,2,4,4]' model.backbone_latent_channels=256 \
  'model.attn_resolutions=[8]' model.decoder_extra_residual_layers=0 \
  model.use_mid_attention=true model.dropout=0.0 \
  model.num_embeddings=2048 model.embedding_dim=256 model.sparsity_level=2 \
  model.patch_based=false model.patch_size=1 model.patch_stride=1 \
  model.recon_mse_weight=0.25 model.recon_l1_weight=1.0 model.recon_edge_weight=0.0 \
  model.perceptual_weight=0.1 model.lpips_version=v0.1 \
  model.perceptual_start_step=0 model.perceptual_warmup_steps=10010 \
  model.adversarial_weight=0.75 model.adversarial_start_step=10010 \
  model.adversarial_warmup_steps=40040 model.adversarial_start_recon_mse=null \
  model.adversarial_quality_ema_decay=0.99 model.disc_start_step=10010 \
  model.disc_learning_rate=4.0e-5 model.discriminator_beta1=0.5 model.discriminator_beta2=0.9 \
  model.disc_num_layers=2 model.disc_channels=64 model.disc_norm=batch \
  model.disc_spectral=false model.disc_loss=hinge \
  model.use_adaptive_disc_weight=true model.disc_weight_max=1.0 \
  model.dead_atom_revival=true model.dead_atom_revival_interval=500 \
  model.dead_atom_revival_max_fraction=0.05 model.dead_atom_revival_noise=0.05 \
  model.dead_atom_revival_patience=5 model.compute_fid=false model.log_images_every_n_steps=500 \
  init_ckpt_path="$clean_ckpt" \
  checkpoint.monitor=val/loss checkpoint.mode=min checkpoint.save_top_k=3 \
  checkpoint.save_last=true checkpoint.every_n_epochs=1 checkpoint.upload_to_wandb=true \
  checkpoint.upload_every_n_epochs=1 +checkpoint.upload_mode=files \
  wandb.project=laser wandb.name="imagenet-rqvae-a2048k2-adv10-fixed-${stamp}" \
  wandb.group=imagenet-rqvae-a2048k2-adv10-fixed wandb.append_timestamp=false \
  wandb.save_dir="$root/wandb" \
  'wandb.tags=[stage1-adv,imagenet,laser,a2048,k2,pixel-dominant,l1-1.0,mse-0.25,lpips-0.1,lpips-v0.1,lpips-eval-locked,gan-start-epoch1,adversarial-0.75,adaptive-cap-1,raw-disc,batchnorm,image-logging,effective-batch128]'
