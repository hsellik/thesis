package Tokenizer.FeaturesEntities;

import com.github.javaparser.ast.body.MethodDeclaration;

public class ExtractedMethod {
    private MethodDeclaration methodDeclaration;
    private String label;

    public ExtractedMethod(MethodDeclaration methodDeclaration, String label) {
        this.methodDeclaration = methodDeclaration;
        this.label = label;
    }

    public MethodDeclaration getMethodDeclaration() {
        return methodDeclaration;
    }

    public String getLabel() {
        return label;
    }
}
