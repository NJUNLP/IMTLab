# IMTLab
IMTLab is an end-to-end interactive machine translation (IMT) evaluation platform that enables researchers to quickly build IMT systems with state-of-the-art models, perform an end-to-end evaluation, and diagnose the weakness of systems.

## Requirements
- Python version >= 3.8.0
- Pytorch version >= 1.8.0
- editdistance
- sentencepiece
- flask
- openai

We include [fairseq](https://github.com/facebookresearch/fairseq) toolkit as a third party library to implement common IMT systems. We recommend users to install fairseq 0.12.2-release from our repository.
``` bash
cd IMTLab
pip install -e third_party/fairseq
```

## Running IMTLab
First change your configures in config/*.json and src/run.sh. The config/\*.json files contain the parameters of the IMT models based on fairseq. And src/run.sh contains the arguments of the environment such as the model type, the policy type, the path of test data, etc. After configuration, simply run the following code.

``` bash
cd src
bash run.sh
```

If you want to use the human environment, change the policy type to 5 in src/run.sh and run the above code.

## Data
In data/ dir is the test data we used in the experiments which is randomly sampled from the testset of WMT.

The human_exps directory contains the human interaction data collected from our human experiments.