# Pytorch UI2code
Pytorch implementation of UI2code.https://github.com/ccywch/UI2code.
# UI2code
UI2code is a tool to convert the UI design image into the skeleton code with deep learning methods.
For example, given an input of Android UI design image, it can generate the corresponding Android XML skeleton code to developers. And developers just need to fill in some detailed attributes such as color, text.
Some brief introduction can be seen in http://tagreorder.appspot.com/ui2code.html

# Paper
ICSE'18 paper:

    From UI Design Image to GUI Skeleton: A Neural Machine Translator to Bootstrap Mobile GUI Implementation
    Chunyang Chen, Ting Su, Guozhu Meng, Zhenchang Xing, Yang Liu
    The 40th International Conference on Software Engineering, Gothenburg, Sweden.
    http://ccywch.github.io/chenchunyang.github.io/publication/ui2code.pdf


# Dataset
It can be downloaded in https://drive.google.com/open?id=17cRSdNPd7GoNuirE983S467kWbOLtiuw and decompressed it for using.

The automated GUI testing tool for android apps, named [Stoat](https://tingsu.github.io/files/stoat.html), to fully-automatically collect UI dataset. Stoat is easy and open to use. 

# Prerequsites

- Python3.6+
- PyTorch 1.4
- CUDA9.2 or higher
- opencv
- tqdm
- perl


# Usage

## Data

The format of `data_path` shall look like:

```
<img_name1> <label_idx1>
<img_name2> <label_idx2>
<img_name3> <label_idx3>
...
```
where `<label_idx>` denotes the line index of the label (starting from 0).
The training data as `train.lst`, validation data as `validate.lst`, testing data as `test_shuffle.lst`.
The raw image data is in `processedImage`, and all code data in `XMLsequence.txt` with the vocabulary as `xml_vocab.txt`.

## Train the Model

All parameters you can adjust in utils/args.py. You can train the model with the following code:
```
python train.py \
--data_base_dir dataset/processedImage \
--data_path dataset/train.lst \
--val_data_path dataset/validate.lst \
--label_path dataset/XMLsequence.lst \
--vocab_file dataset/xml_vocab.txt \
--batch_size 60 \
--beam_size 5 \
--dropout 0.2 \
--num_epochs 10
```
In the default setting, the log file will be put to `log.txt`. The log file records the training and validation perplexities. `model_dir` speicifies where the models should be saved. Please fine-tune the parameter for your own purpose.

## Test the Model

After training, you can load a model and use it to test on test dataset. 

Now you can load the model and test on test set. 
You can use the following code to test it:

```
python translate.py \
--model_path checkpoints/UIModel__step_80000.pth \
--log_path log_test.txt \
--translate_log translate_log.txt \
--data_base_dir dataset/processedImage/ \
--data_path dataset/validate.lst \
--label_path dataset/XMLsequence.lst \
--vocab_file dataset/xml_vocab.txt \
--output_dir results \
--max_num_tokens 100 \
--batch_size 60 --beam_size 5
```

## Evaluate
The test perplexity can be obtained after testing is finished. In order to evaluate the exact match and BLEU, the following command needs to be executed.

```
cd evaluate
python checkExeperiment.py
```
Note that if you change the directory of the results data, please change it accordingly in the file.


# Acknowledgments
This work heavily depends on the <https://github.com/ccywch/UI2code>, <https://github.com/OpenNMT/OpenNMT-py>, thanks for their work.
