from pathlib import Path
from itertools import product
import argparse
import multiprocessing as mp
import numpy as np

import sys
sys.path.append('.')
from src.greedy_routing_annealer import GreedyRoutingAnnealer

parser = argparse.ArgumentParser()
parser.add_argument('--experiments_path', type=Path, required=True)
parser.add_argument('--workers', type=int, default=40)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--ensemble_size', type=int, default=16)
args = parser.parse_args()

graph_names = [
    'polbooks',
    'pso_128',
    'pso_256',
    'metabolic',
    'pso_512',
    'unicodelang',
    'a_song_of_ice_and_fire',
    'pso_1024'
]
embedding_names = ['mercator', 'random']

model_names = [
    'RandomSampling',
    'DegreeDependentSampling',
    'SourceDependentSampling',
    'TargetDependentSampling'
]
experiments_path = args.experiments_path

def run_experiment(experiment_parameters):
    graph_name, _, embedding_name, model_name = experiment_parameters
    annealer = GreedyRoutingAnnealer.new(
        graph_name=graph_name,
        model_name=model_name,
        embedding_name=embedding_name,
        root=experiments_path
    )
    N = len(annealer.coords)
    annealer.embed(N*args.epochs, N, silent=True)

exp_iter = list(product(graph_names, range(args.ensemble_size), embedding_names, model_names))
np.random.shuffle(exp_iter)

with mp.Pool(processes=args.workers) as pool:
    pool.map(run_experiment, exp_iter)