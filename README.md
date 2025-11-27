### activate venv
- `python -m venv venv`
- `source venv/Scripts/activate`

### install dependencies
- `pip install -r requirements.txt`

### run microsoft trocr on sample image
- `python src/main.py`
- pass a custom image with `python src/main.py --image path/to/image.jpeg`
- change model if needed: `python src/main.py --model microsoft/trocr-base-stage1`
