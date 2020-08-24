# Bugram Implementation #

This is an open-source implementation of Bugram by Wang et al.
```
@inproceedings{wang2016bugram,
  title={Bugram: bug detection with n-gram language models},
  author={Wang, Song and Chollak, Devin and Movshovitz-Attias, Dana and Tan, Lin},
  booktitle={Proceedings of the 31st IEEE/ACM International Conference on Automated Software Engineering},
  pages={708--719},
  year={2016}
}
```  
The core idea of the approach is to tokenize Java code from AST using method calls and important nodes from control 
flow such as constructors and initializers; if/else branches; for/do/while/foreach loops; break/continue statements; 
try/catch/finally blocks; return statements; synchronized blocks; 
switch statements and case/default branches, and assert statements. 

Two different n-gram models are created from these tokens and the sequences with the lowest probabilities are extracted. 
To filter out false bugs, the number of reported bugs is reduced by keeping only token sequences at the bottom of at 
two ranked lists generated by different n-gram models with same n-gram size but with different sequence lengths. 

The major difference with the original paper is that this version disregards some AST nodes and also does not 
solve the method calls to their fully qualified names (e.g. methodA() -> org.example.Foo.methodA()). 
Read more in Tokenizer Readme file.

## Configuration
Changing hyper-parameters is possible by editing the file [config.py](config.py).

Here are some of the parameters and their descriptions:
#### config.GRAM_SIZE = 3
The size of an n-gram model.
#### config.SHORTER_SEQUENCE_LEN = 3
The length of the shorter token sequences to be considered when detecting bugs. If sequence is longer than this value, 
it will be broken into chunks with length of this value. Later unioned with longer sequence model results.
#### config.LONGER_SEQUENCE_LEN = 5
The length of the longer token sequences to be considered when detecting bugs. 
Later unioned with shorter sequence model results.
#### config.REPORTING_SIZE = 500
The number of sequences, in the bottom of the ranked list, which will be reported as bugs for both sequences. 
The final bug reports will be a union of the bottom n reported bugs, which is considerably less than reporting size.
#### config.MINIMUM_TOKEN_OCCURENCE = 3
The minimum number of times a token must occur in the software to be included in an n-gram model. Otherwise it is 
simply removed from the token sequences.

## Evaluating a Java project  
* Change `PROJECT_DIR` in [preprocess.sh](preprocess.sh)
* Run `sh preprocess.sh` - this will create file with preprocessed tokens
* Run `sh fit.sh` - this will remove infrequent tokens and fit the n-gram model
* Run `sh evaluate.sh` - this will report a union of two n-gram models with different sequence length
  
Tested only on OS X, but should also work on other Unix systems.  
Windows users can run the java and python manually or create .bat scripts.