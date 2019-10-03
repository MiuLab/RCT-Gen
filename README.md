# RCT-Gen

Code for the paper ["Towards Understanding of Medical Randomized Controlled Trails by Conclusion Generation"](#)  
In Proceedings of the 10th International Workshop on Health Text Mining and Information Analysis at EMNLP ([LOUHI 2019](https://louhi2019.fbk.eu/))

Authors: [Alexander Te-Wei Shieh](https://lipolysis.github.io/), [Yung-Sung Chuang](https://voidism.github.io/), [Shang-Yu Su](https://www.shangyusu.com/), and [Yun-Nung Chen](https://www.csie.ntu.edu.tw/~yvchen/index.html)


## Introduction
Randomized controlled trials (RCTs) represent the paramount evidence of clinical medicine. Using machines to interpret the massive amount of RCTs has the potential of aiding clinical decision-making. We propose a RCT conclusion generation task from the PubMed 200k RCT sentence classification dataset to examine the effectiveness of sequence-to-sequence models on understanding RCTs. We first build a pointer-generator baseline model for conclusion generation. Then we fine-tune the state-of-the-art GPT-2 language model, which is pre-trained with general domain data, for this new medical domain task.
Both automatic and human evaluation show that our GPT-2 fine-tuned models achieve improved quality and correctness in the generated conclusions compared to the baseline pointer-generator model. 
Further inspection points out the limitations of this current approach and future directions to explore.

## Model

We modified the code from [huggingface/pytorch-pretrained-bert](https://github.com/huggingface/transformers) and adjusted the attention mask for fine-tuning on seq2seq data format(from `source` to `conclusion`).

<!-- ![](https://i.imgur.com/o5EkmCn.png) -->
![](https://i.imgur.com/zDKjfua.png)

## Requirements
- python3
- torch>=0.4.0
- nltk
- rouge

install them by:
```
pip install -r requirements.txt
```

## Usage

### Fine-tuning from official gpt-2 pretrained weights
```
usage: gpt2_train.py [-h] [--save_model_name SAVE_MODEL_NAME]
                     [--train_file TRAIN_FILE] [--dev_file DEV_FILE]
                     [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE]
                     [--pred_file PRED_FILE] [--example_num EXAMPLE_NUM]
                     [--mode MODE]

optional arguments:
  -h, --help            show this help message and exit
  --save_model_name SAVE_MODEL_NAME
                        pretrained model name or path to local checkpoint
  --train_file TRAIN_FILE
                        training data file name
  --dev_file DEV_FILE   validation data file name
  --n_epochs N_EPOCHS
  --batch_size BATCH_SIZE
  --pred_file PRED_FILE
                        output prediction file name
  --example_num EXAMPLE_NUM
                        output example number, set to `-1` to run all examples
```

### Testing trained model
```
usage: gpt2_eval.py [-h] [--model_name MODEL_NAME] [--dev_file DEV_FILE]
                    [--pred_file PRED_FILE] [--example_num EXAMPLE_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        pretrained model name or path to local checkpoint
  --dev_file DEV_FILE   validation data file name
  --pred_file PRED_FILE
                        output prediction file name
  --example_num EXAMPLE_NUM
                        output example number, set to `-1` to run all examples
```

## Data

We used the [PubMed 200k RCT dataset](https://github.com/Franck-Dernoncourt/pubmed-rct), which was originally constructed for sequential short text classification, with each sentence labeled as `background`, `objective`, `methods`, `results` and `conclusions`.  

We concatenated the **background**, **objective** and **results** sections of each RCT paper abstract as the model input and the goal of the model is to generate the **conclusions**. If hint words is needed, just concatenate the hint words right after the **results** section. The transformed sample csv file can be found in [data/](https://github.com/MiuLab/RCT-Gen/tree/master/data). 

## Citation

Please use the following bibtex entry:

```
@article{shieh2019rctgen,
  title={Towards Understanding of Medical Randomized Controlled Trails by Conclusion Generation},
  author={Shieh, Alexander Te-Wei and Chuang, Yung-Sung and Su, Shang-Yu and Chen, Yun-Nung},
  year={2019}
}
```
