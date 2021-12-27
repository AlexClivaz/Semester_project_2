source ../venv/Scripts/activate

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '0.1' 'False' 'Feature' 'Unique' 'Corr'

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '0.2' 'False' 'Feature' 'Unique' 'Corr'

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '0.4' 'False' 'Feature' 'Unique' 'Corr'

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '0.8' 'False' 'Feature' 'Unique' 'Corr'

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '1.2' 'False' 'Feature' 'Unique' 'Corr'

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '1.6' 'False' 'Feature' 'Unique' 'Corr'

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '3.2' 'False' 'Feature' 'Unique' 'Corr'

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '6.4' 'False' 'Feature' 'Unique' 'Corr'

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '12.8' 'False' 'Feature' 'Unique' 'Corr'

rm -r data/v1.5.2/processed
python GNN_model.py --dataset_args '25.6' 'False' 'Feature' 'Unique' 'Corr'
