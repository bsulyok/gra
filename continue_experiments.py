from pathlib import Path
from src import annealers
import argparse
import multiprocessing as mp
from omegaconf import OmegaConf
from tqdm import tqdm
import pandas as pd
    
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=20)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--annealer_name', type=str, default=None)
args = parser.parse_args()
    
experiments_root = Path('/home/bsulyok/documents/projects/gra/experiments')

def retrieve_experiments_meta():
    meta = pd.read_csv(experiments_root / 'meta.csv', index_col='name')
    data = []
    for name in meta.index:
        conf = OmegaConf.load(experiments_root / 'exp' / str(name) / 'conf.yml')
        data.append([name, conf.get('graph_name'), conf.get('annealer_name'), conf.get('initial_embedding'), conf.get('step'), conf.get('time'), conf.get('size'), conf.get('grs')])
    meta = pd.DataFrame(data, columns = ['name', 'graph_name', 'annealer_name', 'initial_embedding', 'step', 'time', 'size', 'grs'])
    meta = meta.sample(frac=1)
    meta = meta.set_index('name')
    meta.to_csv(experiments_root / 'meta.csv')
    return meta

def continue_experiment(name):
    experiment_path = experiments_root / 'exp' / str(name)
    if experiment_path.exists():
        conf = OmegaConf.load(experiment_path / 'conf.yml')
        if args.annealer_name is None or args.annealer_name == conf.annealer_name:
            with getattr(annealers, conf.annealer_name)(experiment_path) as annealer:
                annealer.embed(args.steps)

def main():
    meta = retrieve_experiments_meta()
    with open(experiments_root / 'progress.tqdm', mode='a') as progress_file:
        with mp.Pool(processes=args.workers) as pool:
            with tqdm(total=len(meta.index), file=progress_file) as pbar:
                for _ in pool.imap_unordered(continue_experiment, meta.index):
                    pbar.update()

if __name__ == "__main__":
    main()