# AdapatableStreets

SUMO and traci version - 1.15.0
Tensorflow - 2.11.0
Gym -  0.2.1
Python - 3.9.7


## Running on Euler

1. Setup the environment
```
module load gcc/8.2.0 python/3.8.5
```

2. Create a python virtual env
```
python -m venv sumo
```
Note that **sumo** here can be replaced with any venv name of your choosing.

3. Activate the venv
```
source activate sumo/bin/activate
```

4. Install packages
```
pip install -r requirements.txt
```

5. Define `SUMO_HOME` to your bashrc file. If you are using a different shell, edit this accordingly. If you choose to install sumo via other means, also change the `SUMO_HOME` location.
```
echo  'export SUMO_HOME=$(python -m site --user-site)/sumo' >> ~/.bashrc
export SUMO_HOME=$(python -m site --user-site)/sumo
```
*NOTE:* You must repeat step 1 and 3 when running code on Euler. 