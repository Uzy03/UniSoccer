IMAGE      := unisoccer
REMOTE     := ujihara@solar.arch.cs.kumamoto-u.ac.jp
REMOTE_PORT := 2222
REMOTE_DIR := /user/arch/ujihara/UniSoccer
SRC        := SoccerNet/
MATCH_DIR  := SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
JSON_PATH  := $(MATCH_DIR)/clip_dataset.json
CKPT_PATH  := checkpoints/pretrained_classification.pth
COMMENTARY_CKPT    := checkpoints/downstream_commentary_all_open.pth
LLM_CKPT           := meta-llama/Meta-Llama-3-8B-Instruct
INSTRUCTION_CONFIG := configs/instruction_explain.json
INSTRUCTION_CSV    := results/instruction_results.csv
BATCH_SIZE := 4
NUM_WORKERS := 0
MAX_SAMPLES := 0
DEVICE     := cuda
GPU        := 0
OUT_CSV    := results/soccernet_results.csv
COMMENTARY_CSV := results/commentary_results.csv

DOCKER_RUN := docker run --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
              -e CUDA_VISIBLE_DEVICES=$(GPU) \
              --shm-size=8g \
              -v $(CURDIR):/workspace \
              -v $(CURDIR)/hf_cache:/root/.cache/huggingface

.PHONY: build run preprocess inference inference_local inference_commentary inference_instruction extract_clips clean

build:
	docker build --force-rm -t $(IMAGE) .

run:
	docker run -it --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
	    --shm-size=8g \
	    -v $(CURDIR):/workspace \
	    -v $(CURDIR)/hf_cache:/root/.cache/huggingface \
	    $(IMAGE)

preprocess:
	python SoccerNet_script/create_clip_dataset.py \
	    --match_dir "$(MATCH_DIR)"

inference:
	$(DOCKER_RUN) $(IMAGE) \
	    python inference/inference_soccernet.py \
	        --json_path "$(JSON_PATH)" \
	        --ckpt_path $(CKPT_PATH) \
	        --batch_size $(BATCH_SIZE) \
	        --max_samples $(MAX_SAMPLES) \
	        --out_csv $(OUT_CSV)

inference_local:
	CUDA_VISIBLE_DEVICES=$(GPU) python inference/inference_soccernet.py \
	    --json_path "$(JSON_PATH)" \
	    --ckpt_path $(CKPT_PATH) \
	    --batch_size $(BATCH_SIZE) \
	    --max_samples $(MAX_SAMPLES) \
	    --out_csv $(OUT_CSV)

inference_commentary:
	CUDA_VISIBLE_DEVICES=$(GPU) python inference/inference_commentary_soccernet.py \
	    --results_csv $(OUT_CSV) \
	    --json_path "$(JSON_PATH)" \
	    --ckpt_path $(COMMENTARY_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(COMMENTARY_CSV) \
	    --device $(DEVICE)

inference_instruction:
	CUDA_VISIBLE_DEVICES=$(GPU) python inference/inference_instruction_soccernet.py \
	    --config $(INSTRUCTION_CONFIG) \
	    --results_csv $(OUT_CSV) \
	    --json_path "$(JSON_PATH)" \
	    --ckpt_path $(COMMENTARY_CKPT) \
	    --llm_ckpt $(LLM_CKPT) \
	    --out_csv $(INSTRUCTION_CSV) \
	    --device $(DEVICE)

extract_clips:
	python SoccerNet_script/extract_clips.py \
	    --results_csv $(COMMENTARY_CSV) \
	    --json_path "$(JSON_PATH)" \
	    --out_dir results/presentation

upload:
	COPYFILE_DISABLE=1 tar --exclude='._*' --exclude='.DS_Store' -cf - "$(SRC)" | \
	    ssh -p $(REMOTE_PORT) $(REMOTE) \
	        "mkdir -p '$(REMOTE_DIR)' && cd '$(REMOTE_DIR)' && tar -xf -"

clean:
	docker image prune -f
	docker builder prune -f
