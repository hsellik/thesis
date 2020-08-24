package JavaExtractor.Mutator;

import JavaExtractor.App;
import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.SearchUtils;
import JavaExtractor.FeaturesEntities.ExtractedMethod;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.BinaryExpr;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import static JavaExtractor.Common.SearchUtils.containsOffByOne;

public class OffByOneMutator implements MethodMutator {
    CommandLineValues m_CommandLineValues;
    final int BUGGY_METHODS_PER_HUNDRED = 10;
    private Random randomGenerator = new Random();

    public OffByOneMutator(CommandLineValues m_CommandLineValues) {
        this.m_CommandLineValues = m_CommandLineValues;
    }

    /**
     * @param method MethodDeclaration from JavaParser library
     * @return mutations of the method plus the original version
     */
    public List<ExtractedMethod> processMethod(MethodDeclaration method) {
        List<ExtractedMethod> extractedMethods = new ArrayList<>();

        /*List<BinaryExpr> offByOneMutationCandidates = method.getNodesByType(BinaryExpr.class).stream()
                .filter(containsOffByOne())
                .collect(Collectors.toList());

        for (BinaryExpr mutationCandidate : offByOneMutationCandidates) {
            ExtractedMethod extractedMethod = createMutationAndRevert(method, mutationCandidate);
            extractedMethods.add(extractedMethod);
            if (mutatedNodes.indexOf(extractedMethod.getLabel()) == -1) {
                mutatedNodes.append(extractedMethod.getLabel()).append("#");
            }
        }

        if  (extractedMethods.size() != 0) {
            extractedMethods.add(new ExtractedMethod(method, mutatedNodes.toString(), App.noBugString));
        }*/

        List<BinaryExpr> mutationCandidates = method.getNodesByType(BinaryExpr.class).stream()
                .filter(containsOffByOne())
                .collect(Collectors.toList());

        if (mutationCandidates.size() != 0) {
            int index = randomGenerator.nextInt(mutationCandidates.size());
            BinaryExpr mutationCandidate = mutationCandidates.get(index);
            // If enabled, simulate realistic bug ratio
            if (!m_CommandLineValues.RealisticBugRatio || ThreadLocalRandom.current().nextInt(0, 100 + 1) < BUGGY_METHODS_PER_HUNDRED) {
                // Mutate and add method
                ExtractedMethod extractedMethod = createMutationAndRevert(method, mutationCandidate);
                extractedMethods.add(extractedMethod);
            }

            // Add original method
            extractedMethods.add(new ExtractedMethod(App.noBugString, "#" , method));
        }

        return extractedMethods;
    }


    /**
     * @param method            reference to the MethodDeclaration object to be edited
     * @param mutationCandidate reference to the BinaryExpression object to be edited
     * @return a new instance of mutated method
     */
    private ExtractedMethod createMutationAndRevert(MethodDeclaration method, BinaryExpr mutationCandidate) {
        String operator = mutateExpression(mutationCandidate);
        ExtractedMethod mutatedMethod = new ExtractedMethod(SearchUtils.getContainingNodeTypeString(SearchUtils.getContainingNodeType(mutationCandidate)),
                                                            operator, method.clone());
        // Mutate back to get unmodified example, because we edited clone and
        // want the original referenced MethodDeclaration to stay unedited
        mutateExpression(mutationCandidate);

        return mutatedMethod;
    }

    private String mutateExpression(BinaryExpr expression) {
        BinaryExpr.Operator operator = expression.getOperator();
        if (operator.equals(BinaryExpr.Operator.GREATER)) expression.setOperator(BinaryExpr.Operator.GREATER_EQUALS);
        else if (operator.equals(BinaryExpr.Operator.GREATER_EQUALS)) expression.setOperator(BinaryExpr.Operator.GREATER);
        else if (operator.equals(BinaryExpr.Operator.LESS_EQUALS)) expression.setOperator(BinaryExpr.Operator.LESS);
        else if (operator.equals(BinaryExpr.Operator.LESS)) expression.setOperator(BinaryExpr.Operator.LESS_EQUALS);
        return operator.name();
    }

    private void printCandidatesNotConsidered(MethodDeclaration method, List<BinaryExpr> candidatesNotConsidered) {
        if (candidatesNotConsidered.size() == 0) return;

        StringBuilder sb = new StringBuilder();
        String lineSeparator = System.lineSeparator();
        for (BinaryExpr binaryExpr: candidatesNotConsidered) {
            sb.append("#############################################").append(lineSeparator);
            sb.append("A new type of containing node found: ").append(SearchUtils.getContainingNodeType(binaryExpr)).append(lineSeparator);
            sb.append("Node").append(lineSeparator);
            sb.append(binaryExpr).append(lineSeparator);
            sb.append("Method").append(lineSeparator);
            sb.append(method).append(lineSeparator);
        }
        try {
            FileWriter fw = new FileWriter(Paths.get(System.getProperty("user.dir"),  "newContainingNodes.txt").toString());
            fw.write(sb.toString());
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
