package Tokenizer;

import Tokenizer.Common.CommandLineValues;
import Tokenizer.FeaturesEntities.MethodTokens;
import Tokenizer.Visitors.BugramVisitor;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;

import java.util.ArrayList;
import java.util.Optional;
import java.util.stream.Collectors;

class TokenExtractor {
    CommandLineValues m_CommandLineValues;

    TokenExtractor(CommandLineValues m_CommandLineValues) {
        this.m_CommandLineValues = m_CommandLineValues;
    }

    public ArrayList<MethodTokens> extractTokens(String code) {
        CompilationUnit methodCompilationUnit = parseFileWithRetries(code);

        if (methodCompilationUnit == null) {
            return new ArrayList<>();
        }

        return methodCompilationUnit.getTypes().stream()
                .flatMap(type -> type.getMethods().stream())
                .map(this::generateTokensForMethod)
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public CompilationUnit parseFileWithRetries(String code) {
        final String classPrefix = "public class Test {";
        final String classSuffix = "}";
        final String methodPrefix = "SomeUnknownReturnType f() {";
        final String methodSuffix = "return noSuchReturnValue; }";

        String content = code;
        JavaParser parser = new JavaParser();
        Optional<CompilationUnit> optionalParsed = parser.parse(content).getResult();
        if (!optionalParsed.isPresent() || optionalParsed.get().getParsed().equals(Node.Parsedness.UNPARSABLE)) {
            // Wrap with a class and method

            content = classPrefix + methodPrefix + code + methodSuffix + classSuffix;
            optionalParsed = parser.parse(content).getResult();
            if (!optionalParsed.isPresent() || optionalParsed.get().getParsed().equals(Node.Parsedness.UNPARSABLE)) {
                // Wrap with a class only
                content = classPrefix + code + classSuffix;
                optionalParsed = parser.parse(content).getResult();
            }
        }

        return (!optionalParsed.isPresent() || optionalParsed.get().getParsed().equals(Node.Parsedness.UNPARSABLE)) ? null : optionalParsed.get();
    }


    private MethodTokens generateTokensForMethod(MethodDeclaration methodDeclaration) {
        ArrayList<String> tokens = new ArrayList<>();
        BugramVisitor bugramVisitor = new BugramVisitor(tokens);
        bugramVisitor.visit(methodDeclaration, null);

        return new MethodTokens(methodDeclaration.getName().toString(), tokens);
    }

}
