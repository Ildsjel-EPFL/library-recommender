from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import optuna
from pathlib import Path
from tqdm.auto import tqdm
import random
from huggingface_hub import login

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")