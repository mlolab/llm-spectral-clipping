

EXP_DIR=your_path_to_experiment_dir
CHECKPOINT_PATH=$EXP_DIR/ckpts/latest/main.pt
CONFIG_PATH=$EXP_DIR/summary.json
OUTPUT_DIR=$EXP_DIR/eval_results
NUM_FEWSHOT=5

SRC_DIR=your_path_to_src_dir

python "$SRC_DIR/eval/run_eval.py" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config_path "$CONFIG_PATH" \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq,lambada_openai,openbookqa,sciq,mmlu \
    --num_fewshot $NUM_FEWSHOT \
    --batch_size 32 \
    --device cuda:0 \
    --output_dir "$OUTPUT_DIR"
