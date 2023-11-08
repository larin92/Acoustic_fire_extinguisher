# Acoustic 🔥Fire🔥 Extinguisher Prediction Model

## Project Overview

This project utilizes the [Acoustic Extinguisher Fire Dataset](https://www.kaggle.com/datasets/muratkokludataset/acoustic-extinguisher-fire-dataset) to predict whether **acoustic wave** with given `features` will be able to extinguish a 🔥flame🔥 of a given `size`. For more information on dataset, `features`, expirements dataset is based on - refer to the [Jupyter notebook](notebook.ipynb).

The model is served using `FastAPI`/`Uvicorn` and can be deployed using [Docker](#docker-setup).

This project is about predicting , more information below.

For an interesting demonstration of the underlying concept (though not directly related to dataset), check out these YouTube videos:
- [Acoustic 🔥Fire🔥 Extinguisher demo 1](https://www.youtube.com/watch?v=uPVQMZ4ikvM)
- [Acoustic 🔥Fire🔥 Extinguisher demo 2](https://www.youtube.com/watch?v=DanOeC2EpeA)

## Dataset Information

For a detailed exploration of the dataset, feature importance analysis, modeling and tuning - refer to the [Jupyter notebook](notebook.ipynb) included in the repository.

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/acoustic-extinguisher-fire-dataset) or the author's [website](https://www.muratkoklu.com/datasets/vtdhnd07.php). We use the dataset file provided in the project's GitHub repository, and the code for obtaining the dataset for `colab` use is included in the notebook. 
> [Dataset Citations](#citations)

----
## Setup Instructions

#### Repository setup

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/larin92/Acoustic_fire_extinguisher.git

cd Acoustic_fire_extinguisher
```

#### Environment Setup

To set up the Python environment and install dependencies using `pipenv`:

```bash
pip install pipenv

pipenv install
```

#### Running training/serving scripts

To run the training script using `pipenv`:
```bash
pipenv run python .\training_script.py
```

To serve the model using `Uvicorn` (without `Docker`):
```bash
pipenv run python .\serve.py
```

#### Docker Setup

To build and run the Docker container:

```bash
docker build -f Dockerfile -t acoustic_fire_extinguisher:01 .

docker run -d --name acoustic_fire_extinguisher -p 8000:8000 acoustic_fire_extinguisher:01
```

To stop container and clean up:

```bash
docker rm $(docker stop acoustic_fire_extinguisher)
```

## Testing with `curl`

You can test the served model using `curl` with the following commands:

- On Unix:

```bash
curl -i -X POST -H "Content-Type: application/json" -d '{"SIZE": 1, "FUEL": "Gasoline", "DISTANCE": 10, "DECIBEL": 72, "AIRFLOW": 0, "FREQUENCY": 1}' http://localhost:8000/predict

curl -i -X POST -H "Content-Type: application/json" -d '{"SIZE": 4, "FUEL": "Kerosene", "DISTANCE": 100, "DECIBEL": 92.5, "AIRFLOW": 8.5, "FREQUENCY": 38}' http://localhost:8000/predict

curl -i -X POST -H "Content-Type: application/json" -d '{"SIZE": 1, "FUEL": "Gasoline", "DISTANCE": 10, "DECIBEL": 109, "AIRFLOW": 4.5, "FREQUENCY": 67}' http://localhost:8000/predict
```

- On Windows (use `cmd`, not `PowerShell`):

```cmd
curl -i -X POST -H "Content-Type: application/json" -d "{\"SIZE\": 1, \"FUEL\": \"Gasoline\", \"DISTANCE\": 10, \"DECIBEL\": 72, \"AIRFLOW\": 0, \"FREQUENCY\": 1}" http://localhost:8000/predict

curl -i -X POST -H "Content-Type: application/json" -d "{\"SIZE\": 4, \"FUEL\": \"Kerosene\", \"DISTANCE\": 100, \"DECIBEL\": 92.5, \"AIRFLOW\": 8.5, \"FREQUENCY\": 38}" http://localhost:8000/predict

curl -i -X POST -H "Content-Type: application/json" -d "{\"SIZE\": 1, \"FUEL\": \"Gasoline\", \"DISTANCE\": 10, \"DECIBEL\": 109, \"AIRFLOW\": 4.5, \"FREQUENCY\": 67}" http://localhost:8000/predict  
```

## Cloud deployment: TBD

----
## Citations

For more information on the dataset and related studies, please refer to the following citations:

1: KOKLU M., TASPINAR Y.S.,  (2021).  Determining the Extinguishing Status of Fuel Flames With Sound Wave by Machine Learning Methods.  IEEE Access, 9, pp.86207-86216, Doi: 10.1109/ACCESS.2021.3088612  
Link: https://ieeexplore.ieee.org/document/9452168 (Open Access)  
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9452168

2: TASPINAR Y.S., KOKLU M., ALTIN M., (2021).  Classification of Flame Extinction Based on Acoustic Oscillations using Artificial Intelligence Methods.  Case Studies in Thermal Engineering, 28, 101561, Doi: 10.1016/j.csite.2021.101561  
Link: https://www.sciencedirect.com/science/article/pii/S2214157X21007243  (Open Access)  
https://www.sciencedirect.com/sdfe/reader/pii/S2214157X21007243/pdf

3: TASPINAR Y.S., KOKLU M., ALTIN M., (2022).  Acoustic-Driven Airflow Flame Extinguishing System Design and Analysis of Capabilities of Low Frequency in Different Fuels.  Fire Technology, Doi: 10.1007/s10694-021-01208-9  
Link: https://link.springer.com/content/pdf/10.1007/s10694-021-01208-9.pdf"
