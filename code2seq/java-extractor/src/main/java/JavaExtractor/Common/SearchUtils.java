package JavaExtractor.Common;

import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.EnclosedExpr;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.expr.UnaryExpr;

import java.util.function.Predicate;

public class SearchUtils {

    public static boolean containsBinaryWithOffByOne(MethodDeclaration method) {
        return method.getNodesByType(BinaryExpr.class).stream()
                .filter(containsOffByOne())
                .toArray().length != 0;
    }

    public static Predicate<BinaryExpr> containsOffByOne() {
        return expr -> expr.getOperator().equals(BinaryExpr.Operator.GREATER) ||
                expr.getOperator().equals(BinaryExpr.Operator.GREATER_EQUALS) ||
                expr.getOperator().equals(BinaryExpr.Operator.LESS) ||
                expr.getOperator().equals(BinaryExpr.Operator.LESS_EQUALS);
    }

    public static Predicate<BinaryExpr> containsNullCheck() {
        return expr -> expr.getOperator().equals(BinaryExpr.Operator.NOT_EQUALS) &&
                expr.getRight() instanceof NullLiteralExpr;
    }

    public static Class<? extends Node> getContainingNodeType(BinaryExpr binaryExpr) {
        Node parent = binaryExpr.getParentNode().get();
        while (parent instanceof BinaryExpr || parent instanceof EnclosedExpr || parent instanceof UnaryExpr) {
            parent = parent.getParentNode().get();
        }
        return parent.getClass();
    }

    public static String getContainingNodeTypeString(Class<? extends Node> clazz) {
        for (ContainingNode containingNode: ContainingNode.values()) {
            if (containingNode.getNodeClass().equals(clazz)) return containingNode.toString();
        }
        return clazz.getTypeName();
    }

    public static Predicate<BinaryExpr> isContainedBy(Class<? extends Node> clazz) {
        return expr -> getContainingNodeType(expr).equals(clazz);
    }
}
