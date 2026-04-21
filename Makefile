IMAGE      := unisoccer
MATCH_DIR  := SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
JSON_PATH  := $(MATCH_DIR)/clip_dataset.json
CKPT_PATH  := checkpoints/pretrained_classification.pth
BATCH_SIZE := 4
OUT_CSV    := inference/soccernet_results.csv

DOCKER_RUN := docker run --rm --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
              -v $(CURDIR):/workspace

.PHONY: build run preprocess inference clean

build:
	docker build --force-rm -t $(IMAGE) .

run:
	docker run -it --gpus all -e NVIDIA_DISABLE_REQUIRE=1 \
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
	        --out_csv $(OUT_CSV)

clean:
	docker image prune -f
	docker builder prune -f
