# MM_sentiment_RNN
This repo implements the paper "An RNN-Based Multimodal Sentiment Analysis: Focusing on\\ Facial Expressions and Multimodal Dynamic Representations" on Tensorflow r1.1

# Prerequisites
* Python 3.4
* [Tensorflow](https://www.tensorflow.org/) >= 1.1
* NumPy
* sklearn
* pickle

# Data

Download the preprocessed audio and video dataset from [here](https://www.dropbox.com/s/87udb6v403g30tt/MOSI.zip?dl=0).
It consists of two files : audio_dataset.pkl and video_sentence_dataset.pkl. Put them in folder train/MOSI/
Both these files contain lists of size :
```
[example, num_frames, feature_size]
```
As mentionned in the paper, we work with 2096 filtered training examples and the features are of size 172 for audio and 289 for video. 

In folder train/MOSI/, you find train.en, the transcriptions of the videos and train.ids.en, the tokenized text according to vocab.en. The three modalities are sorted according to list_filtered.txt (ids of video clips). So the first utterance of list_filtered.txt has id ```_dI--eQ6qVU_1```, its transcription is the first line of train.en, its audio features at index 0 of audio_dataset.pkl and its video_features at index 0 of video_sentence_dataset.pkl.

We now explain how to train a model. First generate a random order to shuffle the dataset for 10-fold:
```
python3 shuffle.py
```
This create the file shuffle.npy in train/MOSI/ that keeps the shuffle order that will be used throughout the whole training. (its a numpy array of size 2096 eg. [128, 3, 124, 52, ..., 514]). The whole dataset is shuffled and is therefore not speaker independant.

To compute feature-wise accuracy and f1-scores, launch :
```
python3 MOSI_rnn.py --mode [text, audio, video]
```
This computes the modality accuracy and f1-scores on the dataset according to the shuffle.npy order. Each k-fold model generates the {text / audio / video} representation for each utterance, later used for the multi-modality 10-fold. These representation are created in train/MOSI/repr/ with the name ```sentences_repr_{text / audio / video}_kfold{0-9}.npy```.

Once the three modalities computed, you can start the multi-modal fusion
```
python3 FC.py
``` 
Multi-modal fusion k-fold is done according to shuffle.npy (same train-dev-test set for k-fold) so test-set has been never seen even during mono-modalities computations.

You should be able to reproduce (more or less) these results :


| Modality        | Accuracy           | F1-Score  |
| ------------- |:-------------:| -----:|
| Text      | 71.07% | 70.17% |
| Audio     | 60.76%     |   63.87% |
| Video [AU, L, 𝚫L]     | 54.16%      |    49.81% |
| Text + Audio +Video     | 74.16%      |    75.22% |

Further results with speaker independant tests (clips from W8NXH0Djyww_1 to ZUXBRvtny7o_34 of ```list_filtered.txt``` are unseen speakers reserved for test-set - this corresponds to 210 clips or 10.01% of the dataset)

| Modality        | Accuracy           | F1-Score  |
| ------------- |:-------------:| -----:|
| Text      | 66.57% | 69.10% |
| Audio     | 56.66%     |   55.61% |
| Video [AU, L, 𝚫L]     | 51.76%      |    51.79% |
| Text + Audio +Video     | 69.04%      |    71.11% |


