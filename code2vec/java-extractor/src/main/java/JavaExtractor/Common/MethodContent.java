package JavaExtractor.Common;

import com.github.javaparser.ast.Node;

import java.util.ArrayList;

public class MethodContent {
    private final ArrayList<Node> leaves;
    private final String name;
    private final Node rootNode;

    public MethodContent(Node rootNode, ArrayList<Node> leaves, String name) {
        this.rootNode = rootNode;
        this.leaves = leaves;
        this.name = name;
    }

    public ArrayList<Node> getLeaves() {
        return leaves;
    }

    public Node getRootNode() {
        return this.rootNode;
    }

    public String getName() {
        return name;
    }
}
