import numpy as np
import networkx as nx
from .base_annealer import BaseAnnealer
import time
from tqdm import tqdm
    
class ExtraMetrics(BaseAnnealer):
    def __init__(self, experiment_path):
        super().__init__(experiment_path)
        
        self.G = nx.Graph()
        u, v = np.where(self.adjacency_matrix)
        self.G.add_edges_from(np.stack([u, v]).T)
        self.nonadjacent = np.where(np.triu(~self.adjacency_matrix, k=1))


    def geometrical_congruence(self):
        reference_distance = np.arccosh(self.distance)
        cong_sum = 0
        for i, j in zip(*self.nonadjacent):
            paths = nx.all_shortest_paths(self.G, i, j)
            ptsp = np.mean(np.sum([reference_distance[path[:-1], path[1:]] for path in paths], axis=1))
            rd = reference_distance[i, j]
            cong_sum += rd / ptsp
        return cong_sum / len(self.nonadjacent[0])
    
    def greedy_routing_efficiency(self):
        distance = np.arccosh(self.distance)
        gcn = self.gcn
        gd = self.gd
        eff_sum = 0

        for j, i in zip(*self.nonadjacent):
            if gd[i, j] == j:
                reference_distance = distance[i, j]
                grp_distance = 0
                v = i
                while v != j:
                    u = gcn[v,j]
                    grp_distance += distance[u, v]
                    v = u
                eff_sum += reference_distance / grp_distance

        for j, i in zip(*self.nonadjacent):
            if gd[i, j] == j:
                reference_distance = distance[i, j]
                grp_distance = 0
                v = i
                while v != j:
                    u = gcn[v,j]
                    grp_distance += distance[u, v]
                    v = u
                eff_sum += reference_distance / grp_distance
        return eff_sum / len(self.nonadjacent[0]) / 2
    
    def embed(self, steps):
        steps = int(steps)
        if self.conf.grs == 1.0:
            return
            
        log_freq = steps // 100
        log = []
        start = time.time()
        with open(self.experiment_path / 'progress.tqdm', mode='a') as progress_file:
            for step in tqdm(range(steps), file=progress_file, miniters=steps//100):
                self.step += 1
                self.moved_vertex = self.select_for_move()
                self.new_coord = self.random_move()
                self.new_distance = self.recalc_distance()
                self.new_gcn = self.recalc_greedy_closest_neighbour()
                self.new_gd = self.recalc_greedy_destination()
                self.new_grs = self.greedy_routing_score(self.new_gd)
                self.temp = 1 / self.step / self.N
                self.update()
                if step % log_freq == 0:
                    log.append([self.grs, self.geometrical_congruence(), self.greedy_routing_efficiency()])
                if self.grs == 1:
                    self.experiment_path.joinpath('terminated').touch()
                    break
        self.time += time.time() - start
        self.save_experiment(log)