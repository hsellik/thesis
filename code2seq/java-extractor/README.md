# JavaExtractor :file_folder:
The JavaExtractor sub-project is meant to create features to Code2Seq and Code2Vec models out of Java methods. This includes:
* Processing the original Java projects  
* Extracting methods from Java files 
* Mutating the off-by-one situations (if there is none, method is discarded)
* Outputting features to Code2Seq and Code2Vec in the following format:  
`label start_terminal,context_path,end_terminal start_terminal,context_path,end_terminal ...,...,...`
* The exact format depends whether `--code2seq` or `--code2vec` flag is passed to the JAR (see usage example)

### Creating the JAR
* Install [Gradle](https://gradle.org/)
* Run `./gradlew build` inside project root
* `JavaExtractor-0.0.1-SNAPSHOT` will be created in `build/libs/`

### Usage Example
JAR must be already created in order to continue (see above)

#### Generating data for training
* `java -cp path/to/JavaExtractor-0.0.1-SNAPSHOT.jar JavaExtractor.App 
--code2seq true --max_path_length 8 --max_path_width 2 
--dir path/to/training/Java/projects/ --off_by_one true > java-training.txt`
* This command will have labels in the first column for training
* Replace `code2seq` with `code2vec` to get the respective features

#### Generating data for evaluating
*  `java -cp JavaExtractor/JPredict/build/libs/JavaExtractor-0.0.1-SNAPSHOT.jar JavaExtractor.App \
 --code2seq true --max_path_length 8 --max_path_width 2 --dir path/to/project --evaluate true \
 --off_by_one true  > data/evaluate.txt`

* This command will generate context paths with file and method name in first column in order
to find the faulty method when feeding the context paths to the models
* Replace `code2seq` with `code2vec` to get the respective features

### Possible Command Line Values

##### --code2seq false
Enable to output features for code2seq model
##### --code2vec false
Enable to output features for code2vec model
#### --file ""
Path of a single file to be extracted
#### --dir ""
Path of the directory to be extracted
#### --max_path_length 
Max length of the paths
#### --max_path_width
Max width of the paths
#### --evaluate
Evaluation mode, only return file paths as tags
instead of "bug" / "nobug"
#### --realistic_bug_ratio
Add less buggs (1:10) to simulate a more realistic scenario
#### --num_threads
Number of threads to use while extraction
#### --min_code_len
Minimal code length to analyze
#### --max_code_len
Maximum code length to analyze
#### --max_file_len
Maximum file length to analyze
#### --pretty_print
Enable pretty print
#### --max_child_id
#### --off_by_one
Mutate methods to produce off-by-one errors and add them to the output tagged as "bug". 
Discard methods which do not contain binary expressions.
#### --nullpointer
Mutate methods to produce nullpointer errors and add them to the output tagged as "bug". 
Discard methods which do not contain opportunity for nullpointer exception.
#### --only_off_by_one_features
Only outputs paths with nodes containing binary expressions. Other paths do not contain 
paths that go through buggy code for off-by-one errors.
#### --get_json
If true, return paths and AST of a Java method in JSON format