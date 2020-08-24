package Tokenizer.Mutator;

import Tokenizer.Common.CommandLineValues;
import Tokenizer.FeaturesEntities.ExtractedMethod;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.BinaryExpr;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public class OffByOneMutator implements MethodMutator {
    CommandLineValues m_CommandLineValues;
    final int BUGGY_METHODS_PER_HUNDRED = 10;
    private Random randomGenerator = new Random();

    public OffByOneMutator(CommandLineValues m_CommandLineValues) {
        this.m_CommandLineValues = m_CommandLineValues;
    }

    /**
     * @param method MethodDeclaration from JavaParser library
     * @return mutations of the methods
     */
    public List<ExtractedMethod> processMethod(MethodDeclaration method) {
        List<ExtractedMethod> extractedMethods = new ArrayList<>();

        List<BinaryExpr> mutationCandidates = method.getNodesByType(BinaryExpr.class).stream()
                .filter(containsOffByOne())
                .collect(Collectors.toList());

        if (mutationCandidates.size() != 0) {
            int index = randomGenerator.nextInt(mutationCandidates.size());
            BinaryExpr mutationCandidate = mutationCandidates.get(index);
            // If enabled, simulate realistic bug ratio
            if (!m_CommandLineValues.RealisticBugRatio || ThreadLocalRandom.current().nextInt(0, 100 + 1) < BUGGY_METHODS_PER_HUNDRED) {
                // Mutate and add method
                MethodDeclaration mutatedMethod = createMutationAndRevert(method, mutationCandidate);
                extractedMethods.add(new ExtractedMethod(mutatedMethod, "bug"));
            }
            // Add original method
            extractedMethods.add(new ExtractedMethod(method, "nobug"));
        }

        return extractedMethods;
    }


    private MethodDeclaration createMutationAndRevert(MethodDeclaration method, BinaryExpr mutationCandidate) {
        mutateExpression(mutationCandidate);
        MethodDeclaration mutatedMethod = method.clone();
        // Mutate back to get unmodified example, because
        // want the original referenced MethodDeclaration to stay unedited
        mutateExpression(mutationCandidate);

        return mutatedMethod;
    }

    private String mutateExpression(BinaryExpr expression) {
        BinaryExpr.Operator operator = expression.getOperator();
        if (operator.equals(BinaryExpr.Operator.GREATER)) expression.setOperator(BinaryExpr.Operator.GREATER_EQUALS);
        else if (operator.equals(BinaryExpr.Operator.GREATER_EQUALS))
            expression.setOperator(BinaryExpr.Operator.GREATER);
        else if (operator.equals(BinaryExpr.Operator.LESS_EQUALS)) expression.setOperator(BinaryExpr.Operator.LESS);
        else if (operator.equals(BinaryExpr.Operator.LESS)) expression.setOperator(BinaryExpr.Operator.LESS_EQUALS);
        return operator.name();
    }

    public static Predicate<BinaryExpr> containsOffByOne() {
        return expr -> expr.getOperator().equals(BinaryExpr.Operator.GREATER) ||
                expr.getOperator().equals(BinaryExpr.Operator.GREATER_EQUALS) ||
                expr.getOperator().equals(BinaryExpr.Operator.LESS) ||
                expr.getOperator().equals(BinaryExpr.Operator.LESS_EQUALS);
    }
}
