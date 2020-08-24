package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Common.SearchUtils;
import JavaExtractor.FeaturesEntities.ExtractedMethod;
import JavaExtractor.FeaturesEntities.ProgramFeatures;
import JavaExtractor.Mutator.MethodMutator;
import JavaExtractor.Mutator.NullPointerMutator;
import JavaExtractor.Mutator.OffByOneMutator;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.Callable;

class ExtractFeaturesTask implements Callable<Void> {
    private final CommandLineValues m_CommandLineValues;
    private final Path filePath;
    MethodMutator methodMutator;

    public ExtractFeaturesTask(CommandLineValues commandLineValues, Path path) {
        m_CommandLineValues = commandLineValues;
        this.filePath = path;
    }

    @Override
    public Void call() {
        processFile();
        return null;
    }

    public void processFile() {
        ArrayList<ProgramFeatures> features;
        try {
            if (m_CommandLineValues.Evaluate){
                features = extractSingleFileForEvaluation();
            } else {
                features = extractSingleFile();
            }
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        if (features == null) {
            return;
        }
        String toPrint;
        if (m_CommandLineValues.GetJSON) {
            toPrint = featuresToJSON(features);
        } else {
            toPrint = featuresToString(features);
        }
        if (toPrint.length() > 0) {
            System.out.println(toPrint);
        }
    }

    public ArrayList<ProgramFeatures> extractSingleFileForEvaluation() throws IOException {
        String code;
        if (m_CommandLineValues.MaxFileLength > 0 &&
                Files.lines(filePath, Charset.defaultCharset()).count() > m_CommandLineValues.MaxFileLength) {
            return new ArrayList<>();
        }
        try {
            code = new String(Files.readAllBytes(this.filePath));
            FeatureExtractor featureExtractor = new FeatureExtractor(m_CommandLineValues);

            // get methods
            CompilationUnit parseResults = featureExtractor.parseFileWithRetries(code);
            List<MethodDeclaration> methods = null;
            if (parseResults != null) {
                methods = parseResults.getNodesByType(MethodDeclaration.class);
            }
            if (methods == null || methods.size() == 0) return null;
            ArrayList<ProgramFeatures> features = new ArrayList<>();
            for (MethodDeclaration methodDeclaration : methods) {
                if (m_CommandLineValues.ProcessOffByOne) {
                    // filter potential off-by-ones
                    if (SearchUtils.containsBinaryWithOffByOne(methodDeclaration)) {
                        features.addAll(featureExtractor.extractFeatures(methodDeclaration.toString()));
                    }
                }
                if (m_CommandLineValues.ProcessNullpointer) {
                    features.addAll(featureExtractor.extractFeatures(methodDeclaration.toString()));
                }
            }
            // set feature names to find method location during evaluation
            features.forEach(singleMethodFeatures -> singleMethodFeatures.setName("#" + "code2seq" + "#" + this.filePath.toString() + "#" + singleMethodFeatures.getName()));

            return features;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new ArrayList<>();
    }

    private ArrayList<ProgramFeatures> extractSingleFile() throws IOException {
        String code;
        if (m_CommandLineValues.ProcessOffByOne) {
            methodMutator = new OffByOneMutator(m_CommandLineValues);
        }
        if (m_CommandLineValues.ProcessNullpointer) {
            methodMutator = new NullPointerMutator(m_CommandLineValues);
        }


        if (m_CommandLineValues.MaxFileLength > 0 &&
                Files.lines(filePath, Charset.defaultCharset()).count() > m_CommandLineValues.MaxFileLength) {
            return new ArrayList<>();
        }
        try {
            code = new String(Files.readAllBytes(filePath));
            FeatureExtractor featureExtractor = new FeatureExtractor(m_CommandLineValues);

            if (m_CommandLineValues.GetJSON) {
                return featureExtractor.extractFeatures(code);
            } else {
                List<MethodDeclaration> methods = new ArrayList<>();
                // parseFileWithRetries same for code2sec and code2vec feature extractor
                CompilationUnit parseResults = featureExtractor.parseFileWithRetries(code);
                if (parseResults != null) {
                    methods = parseResults.getNodesByType(MethodDeclaration.class);
                }
                // Mutate found methods
                List<ExtractedMethod> allMethods = new ArrayList<>();
                for (MethodDeclaration method: methods) {
                    allMethods.addAll(methodMutator.processMethod(method));
                }
                // Extract features
                return extractFeaturesForAllMethods(featureExtractor, allMethods);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    private ArrayList<ProgramFeatures> extractFeaturesForAllMethods(FeatureExtractor featureExtractor, List<ExtractedMethod> allMethods) {
        ArrayList<ProgramFeatures> allFeatures = new ArrayList<>();
        for (ExtractedMethod extractedMethod : allMethods) {
            ArrayList<ProgramFeatures> features = featureExtractor.extractFeatures(extractedMethod.getMethod().toString());
            if (m_CommandLineValues.OnlyOffByOneFeatures) {
                for (ProgramFeatures methodFeatures : features) {
                    methodFeatures.removeNonOffByOnePaths();
                }
            }
            features.removeIf(ProgramFeatures::isEmpty);
            features.forEach(feature -> {
                feature.setName(extractedMethod.getContainingNode());
                feature.setOriginalOperator(extractedMethod.getOriginalOperator());
            });
            allFeatures.addAll(features);
        }
        return allFeatures;
    }

    public String featuresToString(ArrayList<ProgramFeatures> features) {
        if (features == null || features.isEmpty()) {
            return Common.EmptyString;
        }

        List<String> methodsOutputs = new ArrayList<>();

        for (ProgramFeatures singleMethodFeatures : features) {
            StringBuilder builder = new StringBuilder();

            String toPrint = singleMethodFeatures.toString();
            if (m_CommandLineValues.PrettyPrint) {
                toPrint = toPrint.replace(" ", "\n\t");
            }
            builder.append(toPrint);


            methodsOutputs.add(builder.toString());

        }
        return Common.join(methodsOutputs, "\n");
    }

    public String featuresToJSON(ArrayList<ProgramFeatures> features) {
        if (features == null || features.isEmpty()) {
            return Common.EmptyString;
        }
        // Get only one method/feature to JSON
        return features.get(0).toJSON();
    }
}
