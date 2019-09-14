# Setting AWS E2C Instance



## Connect to the Server

```

$ cd <.pem file path>
$ chomod 400 <yourfilename.pem>
$ ssh -i "yourfilename.pem" ubuntu@ec2-3-19-2018-10.us-east-2.compute.amazonaws.com

```

## Connect to Jupyter Notebook

```
$ ssh -i <youfilename.pem> -L 8000:localhost:8888 ubuntu@ec2-3-19-2018-10.us-east-2.compute.amazonaws.com
$ source activate pytorch_p36
$ jupyter notebook --no-broswer --port=8888
```




