.PHONY: data_preparation
data_preparation:
	@echo "Data Preparation..."
	sh scripts/data_preparation.sh

.PHONY: train
train:
	@echo "Train..."
	sh scripts/train.sh $(ARGS)

.PHONY: inference
inference:
	@echo "Inference..."
	sh scripts/inference.sh $(ARGS)

.PHONY: setup
setup:
	@echo "Setup..."
	bash scripts/setup.sh

.PHONY: serve
serve:
	@echo "Serving..."
	sh scripts/serve.sh

.PHONY: test
test:
	@echo "Testing..."
	bash scripts/test.sh
