# MEAD: Multi-Armed Approach for Evaluation of Adversarial Examples Detectors
MEAD is a framework for evaluating detectors based on several attack strategies. 

### Current package structure
1. The files with the adversarial samples can be found <a href="https://drive.google.com/drive/folders/1w-Qplh5dwLuKhlv4FmF3qYaLDN69rmDA?usp=sharing">here</a>.
2. The features has been created by executing this <a href="">code</a>. 
```
Package
MEAD/
├── adv_data
│   ├── cifar10
│   └── mnist
├── checkpoints
│   ├── rn-best.pt
│   └── small_cnn-best.pt
├── features
│   ├── fs
│   │   ├── cifar10
│   │   │   └── evaluation
│   │   └── mnist
│   │       └── evaluation
│   ├── kd_bu
│   │   ├── cifar10
│   │   │   └── evaluation
│   │   └── mnist
│   │       └── evaluation
│   ├── lid
│   │   ├── cifar10
│   │   │   └── evaluation
│   │   └── mnist
│   │       └── evaluation
│   ├── magnet
│   │   ├── cifar10
│   │   │   └── evaluation
│   │   └── mnist
│   │       └── evaluation
│   └── nss
│       ├── cifar10
│       │   └── evaluation
│       └── mnist
│           └── evaluation
├── __init__.py
├── mead_evaluation.py
├── mead.yml
├── models
│   ├── __init__.py
│   ├── resnet.py
│   └── small_cnn.py
├── README.md
├── results
├── setup_paths.py
└── utils
    ├── general_utils.py
    ├── __init__.py
    └── plot_utils.py
```

#### Usage

To execute MEAD:
- Create the enviroment for MEAD:
```console
foo@bar:~$ conda create --name mead python=3.8
```
- Activate the enviroment for MEAD:
```console
foo@bar:~$ source activate mead
```
- Install all the required packages:
```console
(mead) foo@bar:~$ pip3 install -r requirements.txt
```
- Launch the test from CLI for CIFAR10:

<code>-d</code>: dataset between <code>cifar10</code> and <code>mnist</code><br/>
<code>-m</code>: detector to evaluate between <code>fs</code>, <code>kd_bu</code>, <code>lid</code>, <code>magnet</code> and <code>nss</code><br/>
<code>-p</code>: plot name to save the ROC<br/>
<code>-dev</code>: device between <code>cuda</code> and <code>cpu</code>
```console
(mead) foo@bar:~$ python mead_evaluation.py -d cifar10 -m kd_bu -p cifar10_kd_bu -dev cuda 
```
#### Enviroment
We run each experiment on a machine equipped with an Intel(R) Xeon(R) 
CPU E5-2623 v4, 2.60GHz clock frequency, and a GeForce GTX 1080 Ti GPU.




