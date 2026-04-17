train:
	python train.py

api:
	uvicorn api.app:app --reload

run:
	python train.py && uvicorn api.app:app --reload

test:
	python -m pytest tests/