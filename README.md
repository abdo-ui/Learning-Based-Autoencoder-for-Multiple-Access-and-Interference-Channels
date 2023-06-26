# Learning-Based Autoencoder for Multiple Access and Interference Channels in Space Optical Communications
A pytorch implementation of Learning-Based Autoencoder for Multiple Access and Interference Channels in Space Optical Communications


## Environment setup
```
# using pip
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```

## Experiments
To Train and test the proposed model with the default hyperparameters:
```
python main.py
```
To run the model with specific parameter values:
```
# run RTN_norm model with code rate 7/28, A = 3.0, and 0.2 fading:
python main.py -k 7 -L 28 -A 3.0 -f True -fs 0.2
```
All available parameters can be listed using the following command:
```
python main.py -h
```
