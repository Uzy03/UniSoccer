IMAGE      := unisoccer
REMOTE     := ujihara@solar.arch.cs.kumamoto-u.ac.jp
REMOTE_PORT := 2222
REMOTE_DIR := /user/arch/ujihara/UniSoccer
SRC        := SoccerNet/
MATCH_DIR  := SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
JSON_PATH  := $(MATCH_DIR)/clip_dataset.json
CKPT_PATH  := checkpoints/pretrained_classification.pth
COMMENTARY_CKPT := checkpoints/downstream_commentary_all_open.pth
BATCH_SIZE := 4
NUM_WORKERS := 0
MAX_SAMPLES := 0
DEVICE     := cuda
OUT_CSV    := results/soccernet_results.csv
COMMENTARY_CSV := results/commentary_results.csv

DOCKER_RUN := docker run --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
              --shm-size=8g \
              -v $(CURDIR):/workspace

.PHONY: build run preprocess inference inference_local inference_commentary clean

build:
	docker build --force-rm -t $(IMAGE) .

run:
	docker run -it --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
	    --shm-size=8g \
	    -v $(CURDIR):/workspace \
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
	python inference/inference_soccernet.py \
	    --json_path "$(JSON_PATH)" \
	    --ckpt_path $(CKPT_PATH) \
	    --batch_size $(BATCH_SIZE) \
	    --max_samples $(MAX_SAMPLES) \
	    --out_csv $(OUT_CSV)

inference_commentary:
	python inference/inference_commentary_soccernet.py \
	    --results_csv $(OUT_CSV) \
	    --json_path "$(JSON_PATH)" \
	    --ckpt_path $(COMMENTARY_CKPT) \
	    --out_csv $(COMMENTARY_CSV) \
	    --device $(DEVICE)

upload:
	COPYFILE_DISABLE=1 tar --exclude='._*' --exclude='.DS_Store' -cf - "$(SRC)" | \
	    ssh -p $(REMOTE_PORT) $(REMOTE) \
	        "mkdir -p '$(REMOTE_DIR)' && cd '$(REMOTE_DIR)' && tar -xf -"

clean:
	docker image prune -f
	docker builder prune -f
