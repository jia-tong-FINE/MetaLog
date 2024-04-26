# MetaLog: Generalizable Cross-System Anomaly Detection from Logs with Meta-Learning

## Project Structure
```
├─approaches  # MetaLog main entrance.
├─config      # Configuration for Drain
├─entities    # Instances for log data and DL model.
├─utils
├─logs        
├─datasets    
├─models      # Attention-based GRU.
├─module      # Anomaly detection modules, including classifier, Attention, etc.
├─outputs           
├─parsers     # Drain parser.
├─preprocessing # Preprocessing code, data loaders and cutters.
├─representations # Log template and sequence representation.
└─util        # Vocab for DL model and some other common utils.
```

## Datasets

We used `3` open-source log datasets, HDFS, BGL and OpenStack. 

| Software System | Description                        | Time Span  | # Messages | Data Size | Link                                                      |
|       ---       |           ----                     |    ----    |    ----    |  ----     |                ---                                        |
| HDFS            | Hadoop distributed file system log | 38.7 hours | 11,175,629 | 1.47 GB   | [LogHub](https://github.com/logpai/loghub)                |
| BGL             | Blue Gene/L supercomputer log      | 214.7 days | 4,747,963  | 708.76MB  | [Usenix-CFDR Data](https://www.usenix.org/cfdr-data#hpc4) |
| OpenStack | OpenStack infrastructure log|N.A.|207,820|58.61MB| [LogHub-OpenStack](https://github.com/logpai/loghub/tree/master/OpenStack)|

## Environment

Please refer to the `requirements.txt` file for package installation.

**Key Packages:**


PyTorch v1.10.1

python v3.8.3

hdbscan v0.8.27

overrides v6.1.0

scikit-learn v0.24

tqdm

regex

[Drain3](https://github.com/IBM/Drain3)

## Preparation

- **Step 1:** To run `MetaLog` on different log data, create a directory under `datasets` folder HDFS, BGL and OpenStack.
- **Step 2:** Move target log file (plain text, each raw contains one log message) into the folder of step 1.
- **Step 3:** Download `glove.6B.300d.txt` from [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/), and put it under `datasets` folder.

## Run
- Run `approaches/MetaLog.py` (make sure it has proper parameters) for bilateral generalization from HDFS to BGL.
- Run `approaches/MetaLog_BH.py` (make sure it has proper parameters) for bilateral generalization from BGL to HDFS.
- Run `approaches/MetaLog_OS.py` (make sure it has proper parameters) for zero-shot generalization toward OpenStack.