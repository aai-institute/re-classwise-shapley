python scripts/fetch_data.py --dataset-name $2
python scripts/preprocess_data.py --dataset-name $2
python scripts/run_experiment.py $1 --dataset-name $2
python scripts/render_plots.py $1 --dataset-name $2