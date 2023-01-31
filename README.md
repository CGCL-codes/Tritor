# Tritor: Detecting Semantic Code Clones by Building Social Network-based Triads Model
Tritor is a scalable semantic code clone detector based on semantically enhanced abstract syntax tree. 
We add the control flow and data flow details into the original tree and regard the enhanced tree as a social network.
Then we build a social network-based triads model to collect the similarity features between the two methods by analyzing different types of triads within the network.
After obtaining all features, we use them to train a machine learning-based semantic code clone detector (i.e., Tritor).

Tritor is divided into four phases: AST Generation and Enhancement, Triads Extraction, Feature Extraction, and Classification.

1. AST Generation and Enhancement: 
The purpose of this phase is to perform static analysis to extract the AST and add the control flow and data flow details to the AST to enrich the semantic information incorporated in the AST. 
The input of this phase is a method and the output is a semantically enhanced AST.
2. Triads Extraction: 
The purpose of this phase is to partition the semantically enhanced AST into different types of triads and group them according to the node types. 
The input of this phase is a semantically enhanced AST and the output is the number of various triads in each group.
3. Feature Extraction: 
The purpose of this phase is to extract the similarity scores of triads in the same group one by one. 
The input of this phase is the triads of two methods and the output is the similarity feature.
4. Classification: 
The purpose of this phase is to determine whether two methods are a clone based on the machine learning model trained in advance. 
The input of this stage is a similarity feature vector of two methods and the output is the result of whether they are a clone or not.

The source code and dataset of Tritor will be published here after the paper is accepted.

# Project Structure  
  
```shell  
Amain  
|-- AST_Generation_and_Enhancement.py     	// implement the AST Generation and Enhancement phase  
|-- Triads_Extraction_and_Feature_Extraction.py     // implement the first two phases:  Triads_Extraction and Feature_Extraction
|-- Classification.py   // implement the Classification phase  
```

### AST_Generation_and_Enhancement.py
- Input: dataset with source codes
- Output: semantically enhanced AST of source codes 
```
python AST_Generation_and_Enhancement.py
```

### Triads_Extraction_and_Feature_Extraction.py
- Input: semantically enhanced AST of source codes
- Output: feature vectors of code pairs 
```
python Triads_Extraction_and_Feature_Extraction.py
```

### Classification.py
- Input: feature vectors of dataset
- Output: recall, precision, and F1 scores of machine learning algorithms
```
python Classification.py
```


# Parameter details of our comparative tools
|Tool            |Parameters                     |
|----------------|-------------------------------|
|SourcererCC	|Min lines: 6, Similarity threshold: 0.7            |
|Deckard      |Min tokens: 100, Stride: 2, Similarity threshold: 0.9 |
|RtvNN       |RtNN phase: hidden layer size: 400, epoch: 25, $\lambda_1$ for L2 regularization: 0.005, Initial learning rate: 0.003, Clipping gradient range: (-5.0, 5.0), RvNN phase: hidden layer size: (400, 400)-400, epoch: 5, Initial learning rate: 0.005, $\lambda_1$ for L2 regularization: 0.005, Distance threshold: 2.56    |
|ASTNN      |symbols embedding size: 128, hidden dimension: 100, mini-batch: 64, epoch: 5, threshold: 0.5, learning rate of AdaMax: 0.002  |
|SCDetector      |distance measure: Cosine distance, dimension of token vector: 100, threshold: 0.5, learning rate: 0.0001 |
|DeepSim      |Layers size: 88-6, (128x6-256-64)-128-32, epoch: 4, Initial learning rate: 0.001, $\lambda$ for L2 regularization: 0.00003, Dropout: 0.75 |
|CDLH      |Code length 32 for learned binary hash codes, size of word embeddings: 100 |
|TBCNN      |Convolutional layer dim size: 300，dropout rate: 0.5, batch size: 10 |
|FCCA      |Size of hidden states: 128(Text), 128(AST), embedding size: 300(Text), 300(AST), 64(CFG) clipping gradient range: (-1.2，1.2), epoch: 50, initial learning rate: 0.0005, dropout:0.6, batchsize: 32|
