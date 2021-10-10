#BI-Pass hinglish translator - OmniLotl

This is the prototype project built to be submitted alongside the idea in the MANTHAN HACKATHAON 2021 under the PSID: INTL-DA-05

##Problem statement
Analysis of Hinglish content

##Team Details
|  Name  |  Designation  |
| ------------ | ------------ |
|  Tushar Saxena  |  Leader  |
|  Sanyam Virmani  |  Member  |
|  Kartik Sharma  |  Member  |

##Requirements
1. `python==3.7.x`
2. `tensorflow==2.6.0`
3. `tensorflow_text`
4. `matplotlib`
5. `numpy`
6. `json`

##How to use this code
the usage is a three step process:
1. generate the normalizer model using `python gen-normalizer.py`
2. generate the translator model using `python gen-translator.py`
3. finally, translate text using `python translate.py "your text"`

**Note**: `translate.py` expects the whole sentence to be the first argument so the sentences are needed to be quoted (because of the spaces)

**Bad Call**:  `python translate.py your text`
**Good Call**: `python translate.py "your text"`

##Approach
The problem of translating hinglish sentences to english is divided into to problems with identical form:
1. #####[Normalization] Remove the variance in the input data through Standardization
Because the phonetics of the english language are extremely irregular, the hinglish sentences tend to use multiple forms of word to represent the same set of words. For example, कब is sometimes written as `kab` and other times as `kb`.
Many such variances exists, but tend to follow some pattern which can be easily exploited by a *Sequence2Sequence RNN Model*.
This is the first pass in the translation process and is called **Normalization**
2. #####[Translation] Translate the normalized text into english
After the text is normalized, the translation problem is a fairly trivial Sequence2Sequence Prediction problem.
Another *Sequence2Sequence RNN Model* can be used to translate the normalized text to english

**Note**: The problem statement requires us to handle repeated punctuation appropriately, and, we have concluded that the **repeated punctuations are a valid POS as they convey a valid sentiment**

##How well does the prototype perform?
The models were trained on just above **100kb of data**
When tested with common senteces like "how r u", it produced results like "are are you". Note that the inaccuracy is directly caused by the *lack of training data*.  As the volume of training data increases the results will improve (potentially, Linearly)