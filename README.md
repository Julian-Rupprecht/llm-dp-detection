# LLM-Based Dark Pattern Detection
This repository contains the code for training a large language model for dark pattern detection.

The model used for training is [roberta-base](https://huggingface.co/FacebookAI/roberta-base) from FacebookAI. The dataset can be downloaded [here](https://www.kaggle.com/datasets/krishuppal/dark-patterns)

## Training
1. Change directory to project root
2. Create virtual environment
   `conda create --name <name> --clone base` or `python -m venv .env` (Windows/Linux)
3. Activate virtual environment
   `conda activate <name>` (Conda) or `.env\Scripts\activate` (Windows), `source .env/bin/activate` (Linux)
4. Install requirements using the following command
`pip install -r requirements.txt`
5. If training is performed on a Linux-based system, change directory separators in path specifications from `\\` to `/`
6. Run `train.py` file


## Results
| Objective | Accuracy | Recall | Precision | F1 |
|-------|-------|-------|-------|-------|
| Deceptive? | 0.9788 | 0.9748 | 0.9831 | 0.9789 |
| Category?  | 0.9576 | 0.6378 | 0.7364 | 0.6703 |
