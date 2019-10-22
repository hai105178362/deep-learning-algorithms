# 11-785 Homework2 Part 2 Guidlines

Name: Junxiao Guo
Andrew ID: junxiaog


### Algorithm Architecture 

- Model: Resnet50 (Slightly different from the standard Resnet 50 with different activation functions and hidden layers.)
- Optimizer: Adam
- Loss Function: Central Loss
- Parameters: Please refer to the `cnn_params.py`

### File description

- `main.py`: the main function to run the training psrocess.
- `resnet.py`: Implementation of Resnet 50 (or 101).
- `cnn_params.py`: The parameters of Resnet architecture and  File paths.
- `tracewritter.py`: Record the train&validation results from every epoch.
- `classification_version1.py`: Create `.csv` file for the classification challenge.
- `verification_version1.py`: Create `.csv` file fo the verification challenge.

### Usage Instruction

#### Train model

```shell
python3 main.py
```

#### Get classification result

```
python3 classfication_version1.py
```

#### Get verification result

```
python3 verification_version1.py
```

