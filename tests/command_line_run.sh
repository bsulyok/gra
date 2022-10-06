GRAPH_NAMES="football pso_128 pso_256 metabolic pso_512 unicodelang crime pso_1024"
MODEL_NAMES="BaseAnnealer SourceDependent TargetDependent CloggingDependent"
COORD_NAMES="random mercator"
ENSEMBLE_SIZE=4
STEPS=1e4
for graph_name in $GRAPH_NAMES; do
	for model_name in $MODEL_NAMES; do
		for coord_name in $COORD_NAMES; do
			python ensemble_gra.py --graph_name $graph_name --model_name $model_name --coord_name $coord_name --steps $STEPS --ensemble_size $ENSEMBLE_SIZE &
		done
	done
done
