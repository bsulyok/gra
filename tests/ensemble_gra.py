import numpy as np
import importlib
from pathlib import Path
from tqdm import tqdm
import argparse

import sys
sys.path.append('../')
from src import annealers

results_path = Path('../results')
progress_path = Path('../progress')

parser = argparse.ArgumentParser()
parser.add_argument('--graph_name', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--coord_name', type=str, required=True)
parser.add_argument('--steps', type=float, required=True)
parser.add_argument('--ensemble_size', type=int, required=True)
args = parser.parse_args()

annealer = getattr(annealers, args.model_name)(args.graph_name, args.coord_name)
fname = f'{args.graph_name}_{args.model_name}_{args.coord_name}'
with open(progress_path / f'{fname}.prog', 'w') as progress_file:
    mean, std = annealer.ensemble_embed(args.steps, args.ensemble_size, file=progress_file)
np.savetxt(results_path / f'{fname}.csv', np.stack([mean, std]))