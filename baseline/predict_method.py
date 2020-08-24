import pickle
from subprocess import Popen, PIPE, STDOUT

if __name__ == '__main__':
    code = """String[] reverseArray(final String[] array) {
        final String[] newArray = new String[array.length];
        for (int index = 0; index <= array.length; index++) {
            newArray[array.length - index - 1] = array[index];
        }
    }"""

    file = open('Input.java', 'w')
    file.write(code)
    file.close()

    # Load classifier
    with open('rf_classifier.pkl', 'rb') as fid:
        clf = pickle.load(fid)

    # Load classifier
    with open('tfidf_vectorizer.pkl', 'rb') as fid:
        vectorizer = pickle.load(fid)

    p = Popen(
        ['java', '-cp', './Tokenizer/build/libs/Tokenizer-0.0.1-SNAPSHOT.jar', 'Tokenizer.App', '--file', 'Input.java'],
        stdout=PIPE, stderr=STDOUT)

    label_dict = {
        1: "bug",
        0: "nobug"
    }
    for example in p.stdout:
        print("-----")
        vectorized_example = vectorizer.transform([str(example).split(" ", 1)[1]]).toarray()
        print(f"Original: {str(example).split(' ', 1)[0]}")
        prediction = clf.predict(vectorized_example)
        print(f"Prediction: {prediction} ({label_dict[int(prediction)]})")
        print("-----")
        print("Method:")
        print(str(example).split(" ", 1)[1])
