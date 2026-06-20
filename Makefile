NAMESPACE   := model-recommender
IMAGE       := model-recommender:latest
DEPLOYMENT  := model-recommender-api

.PHONY: build import restart deploy

## Build the Docker image
build:
	docker build -t $(IMAGE) .

## Import the image into k3s containerd (bypasses Docker daemon on the node)
import:
	docker save $(IMAGE) | sudo k3s ctr images import -

## Rolling-restart the API deployment so pods pick up the new image
restart:
	kubectl rollout restart deployment/$(DEPLOYMENT) -n $(NAMESPACE)
	kubectl rollout status  deployment/$(DEPLOYMENT) -n $(NAMESPACE)

## Full deploy: build → import → restart
deploy: build import restart
