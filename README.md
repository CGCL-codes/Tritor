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
|TBCNN      |Convolutional layer dim size: 300， dropout rate: 0.5, batch size: 10 |
|FCCA      |Size of hidden states: 128(Text), 128(AST), embedding size: 300(Text), 300(AST), 64(CFG) clipping gradient range: (-1.2，1.2), epoch: 50, initial learning rate: 0.0005, dropout:0.6, batchsize: 32|


# Descriptions of used metrics in experiments
|Metrics|  Abbr|  Definition|
|----------------|-------------------------------|-----------------------|
|True Positive    |TP   |#samples correctly classified as clone pairs |
|True Negative    |TN   |#samples correctly classified as non-clone pairs  |
|False Positive   |FP   |#samples incorrectly classified as clone pairs |
|False Negative   |FN   |#samples incorrectly classified as non-clone pairs |
|Precision        |P    | TP/(TP+FP) |
|Recall           |R    | TP/(TP+FN) |
|F-measure        |F1   | 2*P*R/(P+R)|


# RQ: Sensitivity
In the previous subsection, we used a balanced test set (i.e., equal number of cloned pairs and non-cloned pairs) to measure the effectiveness of Tritor and to compare it with our baseline. 
However, the project codes to be detected in real life do not follow this balance exactly. 
Therefore, as the ratio of test set clone pairs to non-clone pairs varies in practice, there may be inconsistencies with the detection performance reported above. 
Consequently, in this section, we investigate the sensitivity of Tritor by conducting experiments on seven test sets with different ratios from the BCB dataset, with the training set as well as the saved models consistent.
The overall size of the test set is 27,000 cloned pairs with 27,000 non-cloned pairs (one-tenth of the full dataset). 
Seven different sets are constructed by censoring the number of clone pairs or non-clone pairs. 
For example, a randomly selected quarter of the test set (i.e., 6,750) clone pairs plus all of the non-clone pairs in the test set construct a set with a 1:4 ratio of clone pairs to non-clone pairs.

![the F1 scores, precision and recall for seven test sets with the ratio of clone to non-clone pairs ranging from 4:1 to 1:4](https://github.com/TritorCodes/Tritor/blob/main/img-storage/Proportion.png)

Figure shows the F1 scores, precision and recall for seven test sets with the ratio of clone to non-clone pairs ranging from 4:1 to 1:4. 
We can see that as the proportion of non-clone pairs increases, the precision and F1 scores become progressively lower. 
%Conversely, as the proportion of non-clone pairs decreases, the precision and F1 scores become higher.
What is noticeable is that regardless of the change in proportion, recall remains a relatively stable score.
This is because Tritor is more effective in searching for clones, more committed to exploring semantic similarities between clone pairs, and more scanning for semantic clones that are difficult to detect, and thus has the ability to maintain a stable recall.
On the other hand, non-clone pairs have more difficult-to-capture properties, have a wide variety of features, have more types and are more difficult to resolve, hence the more non-clone pairs exist, the lower the precision. 
Thus the more non-clone pairs exist, the lower precision will be.
Even so, when the ratio of clone to non-clone pairs is 1:4, Tritor is still able to achieve a high F1 score of 96.92% with only 1% fluctuation, demonstrating that Tritor still has stable performance against variations in the ratio of the test set.

Summary:
As the proportion of non-clone pairs increases, the precision and F1 scores become progressively lower, but recall remains a relatively stable score, demonstrating that Tritor has stable performance against variations in the ratio of the test set.

# The feature importance on BCB dataset
All analyses in RQ5 are run on the GCJ dataset and for this revision we have added experiments on the BCB dataset. 
The results of the experiments show that the distribution of both triads types and node types in the top 100 important features are generally consistent with the results of the experiments on GCJ. 
Only some minor deviations are due to differences in code between datasets. 
The detailed data comparison has been placed in the table below.
The first column of the table shows the Triads types, and the next two columns show the number of occurrences of each type in the top 100 important features on the corresponding dataset.

|Triads type |Top 100 on GCJ|Top 100 on BCB|
|----------------|-------------------|-------------------|
|4-021D|30|27|
|5-021U|13|19|
|6-021C|30|34|
|7-111D|0|0|
|8-111U|11|5|
|9-030T|12|13|
|10-030C|0|0|
|12-120D|4|3|
|13-120U|0|0|
|14-120C|0|0|

From the table above we can see that the triads type appears consistently in the top 100 important features on BCB and GCJ.
Both Type 6-021C and Type 4-021D occur most frequently, followed by Type 5-021U, Type 9-030T and Type 8-111U, and Type 12-120D occurs least frequently.

|Top 100 on GCJ|Top 100 on GCJ|Top 100 on BCB|Top 100 on BCB|
|----------------|-------------------|-------------------|-------------------|
|Node Types|Occurrences|Node Types|Occurrences|
|MemberReference|5|MethodInvocation|5|
|ForStatement|5|MemberReference|5|
|BlockStatement|5|BlockStatement|5|
|BinaryOperation|5|BinaryOperation|5|
|WhileStatement|4|StatementExpression|4|
|StatementExpression|4|Operator|4|
|Operator|4|Modifier|4|
|Modifier|4|MethodDeclaration|4|
|MethodInvocation|4|Literal|4|
|LocalVariableDeclaration|4|Identifier|4|
|IfStatement|4|DecimalInteger|4|
|Identifier|4|CatchClauseParameter|4|
|ForControl|4|VariableDeclarator|3|
|EnhancedForControl|4|String|3|
|VariableDeclarator|3|ReferenceType|3|
|VariableDeclaration|3|LocalVariableDeclaration|3|
|String|3|IfStatement|3|
|ReferenceType|3|FormalParameter|3|
|Literal|3|ForStatement|3|
|DecimalInteger|3|BasicType|3|
|BasicType|3|Assignment|3|
|Assignment|3|TryStatement|2|
|TypeArgument|2|ReturnStatement|2|
|MethodDeclaration|2|Null|2|
|FormalParameter|2|ForControl|2|
|ClassDeclaration|2|ClassCreator|2|
|ClassCreator|2|CatchClause|2|
|Cast|2|Cast|2|
|ArraySelector|2|ArraySelector|2|
|ArrayCreator|2|ArrayCreator|2|
|-|-|WhileStatement|1|
|-|-|VariableDeclaration|1|
|-|-|Boolean|1|


From the table above we can see that the node type appears consistently in the top 100 important features on BCB and GCJ.
The vast majority of node types have occurrences that differ by no more than 1 between the two datasets.
There are only a few node types that have larger deviations but these do not exceed 2 and are within a reasonable range of fluctuation.

The two above-mentioned evidences show that the importance of the features we extracted have similar importance on BCB and GCJ dataset, indicating that our method has universal applicability and can be applied to different datasets.
