package JavaExtractor.Mutator;

import JavaExtractor.App;
import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.FeaturesEntities.ExtractedMethod;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.IfStmt;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

public class NullPointerMutator implements MethodMutator {
    private final int NR_BINARY_EXPRESSIONS = 1;
    CommandLineValues m_CommandLineValues;
    final int BUGGY_METHODS_PER_HUNDRED = 10;

    public NullPointerMutator(CommandLineValues m_CommandLineValues) {
        this.m_CommandLineValues = m_CommandLineValues;
    }
    /**
     * @param method MethodDeclaration from JavaParser library
     * @return mutation of the method plus the original version
     */
    public List<ExtractedMethod> processMethod(MethodDeclaration method) {
        List<ExtractedMethod> extractedMethods = new ArrayList<>();

        Random randomizer = new Random();
        MethodDeclaration methodClone = method.clone();
        // Filter out all if statements with format "if (something != null)"
        List<IfStmt> ifStmts = methodClone.findAll(IfStmt.class).stream()
                .filter(ifStm -> ifStm.getNodesByType(BinaryExpr.class).size() == NR_BINARY_EXPRESSIONS
                        && ifStm.getNodesByType(BinaryExpr.class).get(0).getOperator().equals(BinaryExpr.Operator.NOT_EQUALS)
                        && ifStm.getNodesByType(BinaryExpr.class).get(0).getRight() instanceof NullLiteralExpr)
                .collect(Collectors.toList());

        if (ifStmts.size() != 0) {
            // Select random if statement
            IfStmt ifStmt = ifStmts.get(randomizer.nextInt(ifStmts.size()));
            // This is the then block, cloned to avoid touching the AST (for now)
            BlockStmt thenBlock = ifStmt.getThenStmt().clone().asBlockStmt();
            // We have to manipulate the list the if is in, so let's figure it out:
            ifStmt.getParentNode()
                    // Do we have a parent? (In valid Java there should be one)
                    // Assume it's a BlockStmt (you might want to improve this to do what you want)
                    .map(parent -> ((BlockStmt) parent))
                    // Get the statements (one of these should be the if-statement)
                    .map(BlockStmt::getStatements)
                    .ifPresent(statements -> {
                        // Copy the statements in the then-block next to the if statement.
                        thenBlock.getStatements().forEach(thenStmt ->
                                // Use addBefore to get them in the right order (try addAfter to see why)
                                // Clone the statement we're copying to avoid touching the existing AST.
                                statements.addBefore(thenStmt.clone(), ifStmt));
                        // Remove the if statement. (Try removing this line.)
                        ifStmt.remove();
                    });
            // If enabled, simulate realistic bug ratio
            if (!m_CommandLineValues.RealisticBugRatio || ThreadLocalRandom.current().nextInt(0, 100 + 1) < BUGGY_METHODS_PER_HUNDRED) {
                // Add buggy method
                ExtractedMethod buggyMethod = new ExtractedMethod("IF", "!=null", methodClone);
                extractedMethods.add(buggyMethod);
            }
            // Add original non-buggy method
            extractedMethods.add(new ExtractedMethod(App.noBugString, "IF", method));
        }
        return extractedMethods;
    }
}
