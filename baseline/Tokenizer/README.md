# Tokenizer :file_folder:
The Tokenizer sub-project is meant to
* Process the original Java projects  
* Extract tokens from Java methods located inside given files
* Output method in the following format:  
`label token1 token2 token3 token4 ... sequence_length`
* This project does not filter out rare tokens

#### Generating tokens
* `java -cp build/libs/Tokenizer-0.0.1-SNAPSHOT.jar Tokenizer.App --dir path/to/java/projects > tokens.txt`

### Building the JAR
You can edit the source code and build your own jar
* Install [Gradle](https://gradle.org/)
* Run `./gradlew build`
* `Tokenizer-0.0.1-SNAPSHOT` will be created in `build/libs/`

### Credits
The general structure of the code is copied from 
[Code2Vec JavaExtractor](https://github.com/tech-srl/code2vec/tree/master/JavaExtractor/JPredict/src/main/java/JavaExtractor)