# Vecamend: Word Vectors Amending using Sentiment Lexicons

Vecamend: **Vec**tors **amend**

This project is a experiment for testing our method to amend word vectors obtained from any pre-trained models using sentiment lexicons.

### Chinese Introduction

动机：目前众多神经语言模型提供的词向量，不能很好的反映情感相似性，因为上下文相似的词语并一定具有相似的情感类别。传统的情感词典为情感词提供了重要的信息，词向量训练过程并没有融入传统词典信息，我们希望借助情感词典来优化现有词向量。

方法：基本思想是，利用情感词典（里面有正向词、负向词列表），让正向词尽量靠近正向词中心，并远离负向词中心，让负向词尽量靠近负向词中心，并远离正向词中心，同时不要偏离原始向量太远。具体而言，目标函数为：

<p align="center"> <img src="./images/formula_1.png" height="180" /> </p>

目标是最小化目标函数*J(Θ)*，A部分目的是让正向词尽量靠近正向词中心，并远离负向词中心；B部分目的是让负向词尽量靠近负向词中心，并远离正向词中心；C部分目的是优化后的向量不要偏离原始向量太远。P、Q分别是正向和负向情感词对应的index。公式中A、B、C之间为相乘，也可以是相加，还需进一步确定。

如何求解最优解：使用iterative updating method。

实验论证：单词VA预测、twitter情感分类两个实验。

### Semantic Similar Words May Have Different Sentiment Polarity

Word vectors obtained from neural language models often suffer from the dilemma that semantic similar words may have different sentiment polarity, as these models are based on the assumption that words occurring in similar context tend to have similar word vectors. 


### Paper Reviews

|References|Emotion Model|Level|Lexicon|Features|Method|Word Embeddings|
|------|------|------|------|------|------|------|
|[Kim 2014]|Categorical|Sentence|None|None|CNN|word2vec|
|[Kalchbrenner 2014]|Categorical|Sentence|None|None|DCNN|randomly initialised|
|[Wang 2015]|Categorical|Sentence|None|None|CNN|Senna, GloVe, Word2Vec|
|[Johnson 2015]|Categorical|Sentence|None|None|seq-CNN, bow-CNN|one-hot representation|
|[dos Santos 2014]|Categorical|Sentence|None|None|CharSCNN|Word-Level & Character-Level vectors|
|[Tang 2014]|Categorical|Sentence|None|SSWE & hand-crafted features|SVM|SSWE|
|[Yin 2015]|Categorical|Sentence|None|None|MVCNN|HLBL、Huang、GloVe、SENNA、word2vec|
|[Ruppenhofer 2014]|Dimensional|Word|Warriner, SentiStrength, SoCAL|modifiers, star ratings or lexicon| Corpus-based & lexical-based methods|None|
|[Staiano 2014]|Dimensional|Word, Sentence|Wordnet, DepecheMood|Lexicon|Regression|None|
|[Xu 2015]|Dimensional|Sentence|Opinion Lexicon, Afinn, MPQA, SentiWordnet|Lexicon, n-grams|decision tree regression|None|
|[Van Hee 2015]|Dimensional|Sentence|WordNet|lexical and syntactic features|regression|None|
|[Gimenez 2015]|Dimensional|Sentence|Pattern, Afinn-111, Jeffrey, NRC, SentiWordNet|lexicon, ngrams, negation, syntactic|linear kernel SVR|None|
|[Farıas 2015]|Dimensional|Sentence|AFINN, ANEW, DAL, HL, GI, SWN, SN, LIWC, NRC|Lexical & Structural Features|Linear Regression|None|
|[Gupta 2015]|Dimensional|Sentence|None|bags of words, bags of character 3-grams, binary features|Autoencoders, regression|None|
|[McGillion 2015]|Dimensional|Sentence|None|n-grams, uppercase, punctuations & word embeddings|stacking systems: regression|word2vec|
|[Karanasou 2015]|Dimensional|Sentence|None|syntactical and morphological features|SVM Classification|None|
|[Han 2015]|Dimensional|Sentence|Liu, MPQA, Mohammad|formal text features, tweet-specific features, discourse features, sentiment distribution among topics & word embedding|SVM classifier|SSWE|
|[Nguyen 2015]|Dimensional|Sentence|None|term features and emotion patterns|a decision tree based classifier|None|
|[Dragoni 2015]|Dimensional|Sentence|None|None|IR-based|None|

### References
___
**CNN**

[Kim 2014] Kim, Y. (n.d.). Convolutional Neural Networks for Sentence Classification. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP).

[Kalchbrenner 2014] Kalchbrenner, N., Grefenstette, E., & Blunsom, P. (n.d.). A Convolutional Neural Network for Modelling Sentences. Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

[Wang 2015] Wang, P., Xu, J., Xu, B., Liu, C. L., Zhang, H., Wang, F., & Hao, H. (2015). Semantic Clustering and Convolutional Neural Network for Short Text Categorization. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Vol. 2, pp. 352-357).

[Johnson 2015] Johnson, R., & Zhang, T. (n.d.). Effective Use of Word Order for Text Categorization with Convolutional Neural Networks. Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.

[dos Santos 2014] dos Santos, C. N., & Gatti, M. (2014). Deep convolutional neural networks for sentiment analysis of short texts. In Proceedings of the 25th International Conference on Computational Linguistics (COLING), Dublin, Ireland.

[Tang 2014] Tang, D., Wei, F., Qin, B., Liu, T., & Zhou, M. (2014, August). Coooolll: A deep learning system for twitter sentiment classification. In Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014) (pp. 208-212).

[Yin 2015] Wenpeng Yin, Hinrich Schütze. Multichannel Variable-Size Convolution for Sentence Classification. The 19th SIGNLL Conference on Computational Natural Language Learning (CoNLL'2015, long paper). July 30-31, Peking, China.
___
**SemEval-2015 Task 11**

[Ruppenhofer 2014] Josef Ruppenhofer, Michael Wiegand, Jasper Brandes: Comparing methods for deriving intensity scores for adjectives. EACL 2014: 117-122.

[Staiano 2014] Jacopo Staiano, Marco Guerini: DepecheMood: a Lexicon for Emotion Analysis from Crowd-Annotated News. CoRR abs/1405.1605 (2014).

[Xu 2015] Xu, Hongzhi and Santus, Enrico and Laszlo, Anna and Huang, Chu-Ren 2015, LLT-PolyU: Identifying Sentiment Intensity in Ironic Tweets, In Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval 2015), Association for Computational Linguistics, pages 673–678, Denver, Colorado.
___

**SemEval-2015 Task 11 第四名及其后**

[Van Hee 2015] Van Hee, C., Lefever, E., & Hoste, V. (2015). LT3: sentiment analysis of figurative tweets: piece of cake# NotReally. In SemEval: 9th International Workshop on Semantic Evaluations at Naacl 2015 (pp. 684-688). Association for Computational Linguistics.

[Gimenez 2015] Mayte Gimenez, Ferran Pla, Lluıs-F. Hurtado. ELiRF: A Support Vector Machine Approach for Sentiment Analysis Tasks in Twitter at SemEval-2015. SemEval-2015.

[Farıas 2015] Farıas, D. I. H., Sulis, E., Patti, V., Ruffo, G., & Bosco, C. ValenTo: Sentiment Analysis of Figurative Language Tweets with Irony and Sarcasm.

[Gupta 2015] Gupta, P., & Gómez, J. A. PRHLT: Combination of Deep Autoencoders with Classification and Regression Techniques for SemEval-2015 Task 11.

[McGillion 2015] McGillion, S., Martinez Alonso, H., & Plank, B. CPH: Sentiment analysis of Figurative Language on Twitter# easypeasy# not.

[Karanasou 2015] Karanasou, M., Doulkeridis, C., & Halkidi, M. DsUniPi: An SVM-based Approach for Sentiment Analysis of Figurative Language on Twitter.

[Han 2015] Han, X., Li, B., Ma, J., Zhang, Y., Ou, G., Wang, T., & Wong, K. F. UIR-PKU: Twitter-OpinMiner System for Sentiment Analysis in Twitter at SemEval 2015.

[Nguyen 2015] Nguyen, H. L., Nguyen, T. D., Hwang, D., & Jung, J. J. KELabTeam: A Statistical Approach on Figurative Language Sentiment Analysis in Twitter.

[Dragoni 2015] Dragoni, M., & Povo, T. SHELLFBK: An Information Retrieval-based System For Multi-Domain Sentiment Analysis.
___