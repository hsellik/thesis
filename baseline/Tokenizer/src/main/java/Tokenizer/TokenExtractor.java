package Tokenizer;

import Tokenizer.Common.CommandLineValues;
import Tokenizer.FeaturesEntities.ExtractedMethod;
import Tokenizer.FeaturesEntities.MethodTokens;
import Tokenizer.Mutator.MethodMutator;
import Tokenizer.Mutator.OffByOneMutator;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Optional;
import java.util.stream.Collectors;

class TokenExtractor {
    CommandLineValues m_CommandLineValues;
    JavaParser javaParser;

    TokenExtractor(CommandLineValues m_CommandLineValues) {
        this.m_CommandLineValues = m_CommandLineValues;
        javaParser = new JavaParser();
        javaParser.getParserConfiguration().setAttributeComments(false);
    }

    public ArrayList<MethodTokens> extractTokens(String code) {
        if (m_CommandLineValues.Method) {
            Optional<MethodDeclaration> optionalMethodDeclaration = javaParser.parseMethodDeclaration(code).getResult();
            if (optionalMethodDeclaration.isPresent()) {
                return generateTokensForMethod(optionalMethodDeclaration.get());
            } else {
                return new ArrayList<>();
            }
        } else {
            CompilationUnit fileCompilationUnit = parseFileWithRetries(code);

            if (fileCompilationUnit == null) {
                return new ArrayList<>();
            }

            return fileCompilationUnit.getTypes().stream()
                    .flatMap(type -> type.getMethods().stream())
                    .map(this::generateTokensForMethod)
                    .flatMap(Collection::stream)
                    .collect(Collectors.toCollection(ArrayList::new));
        }
    }

    public CompilationUnit parseFileWithRetries(String code) {
        final String classPrefix = "public class Test {";
        final String classSuffix = "}";
        final String methodPrefix = "SomeUnknownReturnType f() {";
        final String methodSuffix = "return noSuchReturnValue; }";

        String content = code;
        Optional<CompilationUnit> optionalParsed = javaParser.parse(content).getResult();
        if (!optionalParsed.isPresent() || optionalParsed.get().getParsed().equals(Node.Parsedness.UNPARSABLE)) {
            // Wrap with a class and method

            content = classPrefix + methodPrefix + code + methodSuffix + classSuffix;
            optionalParsed = javaParser.parse(content).getResult();
            if (!optionalParsed.isPresent() || optionalParsed.get().getParsed().equals(Node.Parsedness.UNPARSABLE)) {
                // Wrap with a class only
                content = classPrefix + code + classSuffix;
                optionalParsed = javaParser.parse(content).getResult();
            }
        }

        return (!optionalParsed.isPresent() || optionalParsed.get().getParsed().equals(Node.Parsedness.UNPARSABLE)) ? null : optionalParsed.get();
    }


    private ArrayList<MethodTokens> generateTokensForMethod(MethodDeclaration methodDeclaration) {
        MethodMutator mutator = new OffByOneMutator(m_CommandLineValues);
        ArrayList<MethodTokens> methodsTokens = new ArrayList<>();

        for (ExtractedMethod extractedMethod : mutator.processMethod(methodDeclaration)) {
            methodsTokens.add(new MethodTokens(extractedMethod.getLabel(),
                    tokenizeMethod(extractedMethod.getMethodDeclaration())));
        }

        return methodsTokens;
    }

    private ArrayList<String> tokenizeMethod(MethodDeclaration methodDeclaration) {
        ArrayList<String> tokens = new ArrayList<>();
        // Reparse to update tokenrange
        methodDeclaration = javaParser.parseMethodDeclaration(methodDeclaration.toString()).getResult().get();
        if (methodDeclaration.getTokenRange().isPresent()) {
            methodDeclaration.getTokenRange().get()
                    .forEach(token -> {
                        if (!token.getText().equals(" ") && !token.getText().equals("\n")) {
                            tokens.add(token.getText());
                        }
                    });
        }
        return tokens;
    }

}
