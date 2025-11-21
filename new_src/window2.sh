python new_src/window_eval_plot.py \
  --data_file new_results/attacks/saliency_strong_2k.tsv \
  --ckpt checkpoints/run_weighted_w0_8/best.pt \
  --vocab checkpoints/vocab_n.json \
  --out_dir new_results/window_eval_after \
  --events_per_window 16 --stride_events 4 --batch_size 256 --max_len 256
