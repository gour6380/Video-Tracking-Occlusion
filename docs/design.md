# Experimental design

This is a bounded adaptation and engineering study, not a full SAM 2 reproduction. The pretrained model is the control; the selected adapted model is the treatment. Fine-tuning is allowed to produce a negative result.

## Frozen protocol

SAM 2.1 Hiera Tiny is loaded strictly from the creator checkpoint. Only `sam_mask_decoder`, `memory_attention`, `memory_encoder`, `obj_ptr_proj` and `obj_ptr_tpos_proj` parameters train. Frozen encoders stay in evaluation mode. The adapter does not detach temporal masks, memory or trainable decoder projections. A pinned-upstream zero-correction-loop edge case is handled by disabling the empty correction loop rather than adding hidden clicks.

The pinned predictor also references a missing per-object memory-clearing helper. The project adapter supplies that helper on each predictor instance, preserving the upstream temporal radius while clearing only the selected object's non-conditioning memory. Other objects and conditioning prompts are retained; vendored source and its checksums remain unchanged. Native preflight/propagation fixtures and a random-weight CPU tracking smoke exercise both callers. This compatibility fix does not establish CUDA/BF16 correctness or tracking quality.

Four released chronological frames, one object, strides 1/2/4/8, native 1024-square RGB normalization, consistent 0.5-probability horizontal flip. A target must be visible initially; later empty annotations remain supervised. Training box prompts inherit upstream corner jitter (10% of box size, capped at 20 internal pixels). Initial prompt mix is point/box/mask 50/25/25, no future conditioning or corrective training frames. Loss uses upstream focal mask/Dice/IoU/object-presence weights 20/1/1/1 and documented frame-summing reduction, normalized over four accumulation microbatches.

AdamW: 1e-5 peak LR, 1e-6 final, betas .9/.999, epsilon 1e-8, matrix decay .01 with biases/norms excluded, clipping 1.0. Fifty warm-up updates followed by cosine decay; 1,000 total updates, seed 2027. BF16 autocast with FP32 weights and optimizer state. No torch.compile. Checkpoints every 100 updates; development scoring every 200 and final update. Highest development point J&F-dot wins, earlier ties retained. No automatic extra training.

## Data identity and split validity

Author HF revision: `24bd8fc5f5ab824a482b5f6dcf472e3a3ef5c704`. Download the three train archive parts, validation archive and pinned metadata/checksum document. Verify every archive, safely extract regular files, and record per-file identities. A small study subset does not reduce the initial bulk download.

Audit all training/validation frames for exact file duplicates and sample frames for conservative visual matches. Join connected groups before allocating approximately 300/40/60 clips. Whole groups remain intact and groups intersecting official validation are excluded from local partitions. Missing masks are errors, never empty-target examples. Public metadata lacks original-source IDs; exact/near-duplicate audit is an approximation, not a proof of independence. No strict foundation-model pretraining-exclusion claim is made.

## Local metrics and annotation assistance

Use pinned public scorer revision `a74b23aa3a31597f487eec10f452e0b9f61347e1`. Preserve add-one-smoothed J, adaptive boundary F, native geometry, first/last frame exclusions, equal per-object aggregation and upstream disappearance/reappearance event rules. Unsupported event scores are null/N/A, never zero. Pixel metric code is restored unchanged; the wrapper validates completeness and reproduces aggregation with tested temporal exclusions.

One frozen interior point per object initializes click experiments. The assisted protocol offers one click at the predeclared midpoint, selected from the largest false-negative/false-positive connected component; no error means no click. Both paired arms omit that frame, without merging temporal events across excluded frames. Primary unassisted scores retain all otherwise eligible frames. Ground truth assists prompting only at allowed frames, not automatic detection or checkpoint selection on test data.

Full-mask initialization is a separate protocol. Complete official validation exports preserve all 433 clips, frame names, target IDs and native indexed PNG dimensions, with no corrective prompts. Local public-code scores and private server scores remain separate. Only a real imported receipt supports an official result.

## Timing and state

Input images remain in CPU storage. For synchronized model timing, one frame and prompt tensors are staged onto CUDA before the timed call; output materialization, rendering and writes occur afterward. Model loading and upstream session initialization (including the first image encoding) are separately reported. One warm-up and three measured complete trajectories reset states; pilot measurements use one measured trajectory to limit setup work. First-call/trajectory cost is separate. Allocator memory is not process RSS.

Multiple-object state uses persistent IDs and maximum-positive-logit overlap resolution. Each video/checkpoint/session is isolated. An earlier edit rebuilds chronological state; a future correction is not available before its timestamp. Benchmark restoration checks repeated indexed predictions and cleared object IDs.

## Limits and completion

One seed, approximate source grouping, short training windows and differing click/mask protocols limit claims. BF16/CUDA behavior, A40 fit, GPU timings and quality are unverified until the pilot/main notebook execute. No implementation-preparation receipt closes an experimental or publication gate.
