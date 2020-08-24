package Tokenizer.FeaturesEntities;

import java.util.ArrayList;
import java.util.List;

public class MethodTokens {

    private String methodName;
    private ArrayList<String> tokens = new ArrayList<>();

    public MethodTokens(String methodName, List<String> tokens) {
        this.methodName = methodName;
        this.tokens.addAll(tokens);
    }

    public String getMethodName() {
        return methodName;
    }

    public void setMethodName(String methodName) {
        this.methodName = methodName;
    }

    @Override
    public String toString() {
        return methodName + " " + String.join(" ", tokens);
    }
}
