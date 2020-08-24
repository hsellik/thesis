package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Common.MethodContent;
import JavaExtractor.FeaturesEntities.Paths;
import JavaExtractor.FeaturesEntities.ProgramFeatures;
import JavaExtractor.Visitors.FunctionVisitor;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@SuppressWarnings("StringEquality")
class FeatureExtractor {
    // Code2Seq symbols
    private final static String shortSymbol = "|";
    // Code2Vec symbols
    final static String startSymbol = "(";
    final static String endSymbol = ")";
    // Common symbols
    private final static String upSymbol = "^";
    private final static String downSymbol = "_";

    private static final Set<String> s_ParentTypeToAddChildId = Stream
            .of("AssignExpr", "ArrayAccessExpr", "FieldAccessExpr", "MethodCallExpr")
            .collect(Collectors.toCollection(HashSet::new));
    private final CommandLineValues m_CommandLineValues;

    public FeatureExtractor(CommandLineValues commandLineValues) {
        this.m_CommandLineValues = commandLineValues;
    }

    private static ArrayList<Node> getTreeStack(Node node) {
        ArrayList<Node> upStack = new ArrayList<>();
        Node current = node;
        while (current != null) {
            upStack.add(current);
            current = current.getParentNode().isPresent() ? current.getParentNode().get() : null;
        }
        return upStack;
    }

    public ArrayList<ProgramFeatures> extractFeatures(String code) {
        CompilationUnit m_CompilationUnit = parseFileWithRetries(code);
        FunctionVisitor functionVisitor = new FunctionVisitor(m_CommandLineValues);

        functionVisitor.visit(m_CompilationUnit, null);

        ArrayList<MethodContent> methods = functionVisitor.getMethodContents();

        return generatePathFeatures(methods);
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

    private ArrayList<ProgramFeatures> generatePathFeatures(ArrayList<MethodContent> methods) {
        ArrayList<ProgramFeatures> methodsFeatures = new ArrayList<>();
        for (MethodContent content : methods) {
            ProgramFeatures singleMethodFeatures = generatePathFeaturesForFunction(content);
            if (!singleMethodFeatures.isEmpty()) {
                methodsFeatures.add(singleMethodFeatures);
            }
        }
        return methodsFeatures;
    }

    private ProgramFeatures generatePathFeaturesForFunction(MethodContent methodContent) {
        ArrayList<Node> functionLeaves = methodContent.getLeaves();
        ProgramFeatures programFeatures = new ProgramFeatures(methodContent.getRootNode(), methodContent.getName(), this.m_CommandLineValues);

        for (int i = 0; i < functionLeaves.size(); i++) {
            for (int j = i + 1; j < functionLeaves.size(); j++) {
                String separator = Common.EmptyString;

                Paths paths = generatePaths(functionLeaves.get(i), functionLeaves.get(j), separator);
                if (paths != null && paths.getMainPath() != Common.EmptyString) {
                    String mainPath = paths.getMainPath();
                    String longPath = paths.getLongPath();
                    List<Integer> idPath = paths.getIdPath();
                    Node source = functionLeaves.get(i);
                    Node target = functionLeaves.get(j);
                    programFeatures.addFeature(source, mainPath, longPath, idPath, target);
                }
            }
        }
        return programFeatures;
    }

    private Paths generatePaths(Node source, Node target, String separator) {

        StringJoiner stringBuilderMainPath = new StringJoiner(separator);
        StringJoiner stringBuilderLongPath = new StringJoiner(separator);
        ArrayList<Integer> idPath = new ArrayList<>();
        ArrayList<Node> sourceStack = getTreeStack(source);
        ArrayList<Node> targetStack = getTreeStack(target);

        int commonPrefix = 0;
        int currentSourceAncestorIndex = sourceStack.size() - 1;
        int currentTargetAncestorIndex = targetStack.size() - 1;
        while (currentSourceAncestorIndex >= 0 && currentTargetAncestorIndex >= 0
                && sourceStack.get(currentSourceAncestorIndex) == targetStack.get(currentTargetAncestorIndex)) {
            commonPrefix++;
            currentSourceAncestorIndex--;
            currentTargetAncestorIndex--;
        }

        int pathLength = sourceStack.size() + targetStack.size() - 2 * commonPrefix;
        if (pathLength > m_CommandLineValues.MaxPathLength) {
            return null;
        }

        if (currentSourceAncestorIndex >= 0 && currentTargetAncestorIndex >= 0) {
            int pathWidth = targetStack.get(currentTargetAncestorIndex).getData(Common.ChildId)
                    - sourceStack.get(currentSourceAncestorIndex).getData(Common.ChildId);
            if (pathWidth > m_CommandLineValues.MaxPathWidth) {
                return null;
            }
        }

        for (int i = 0; i < sourceStack.size() - commonPrefix; i++) {
            Node currentNode = sourceStack.get(i);
            String childId = Common.EmptyString;
            String parentRawType = currentNode.getParentNode().get().getData(Common.PropertyKey).getRawType();
            if (i == 0 || s_ParentTypeToAddChildId.contains(parentRawType)) {
                childId = saturateChildId(currentNode.getData(Common.ChildId))
                        .toString();
            }

            if (m_CommandLineValues.Code2Seq) {
                stringBuilderMainPath.add(String.format("%s%s%s",
                        currentNode.getData(Common.PropertyKey).getType(true), childId, shortSymbol));
            } else if (m_CommandLineValues.Code2Vec) {
                stringBuilderMainPath.add(String.format("%s%s%s%s%s", startSymbol,
                        currentNode.getData(Common.PropertyKey).getType(false), childId, endSymbol, upSymbol));
            }
            if (m_CommandLineValues.GetJSON) {
                stringBuilderLongPath.add(String.format("(%s%s)%s",
                        currentNode.getData(Common.PropertyKey).getType(false), childId, upSymbol));
                idPath.add(currentNode.getData(Common.NodeId));
            }

        }

        Node commonNode = sourceStack.get(sourceStack.size() - commonPrefix);
        String commonNodeChildId = Common.EmptyString;
        Optional<Node> parentNode = commonNode.getParentNode();
        String commonNodeParentRawType = Common.EmptyString;
        if (parentNode.isPresent() && parentNode.get().containsData(Common.PropertyKey)) {
            commonNodeParentRawType = parentNode.get().getData(Common.PropertyKey).getRawType();
        }
        if (s_ParentTypeToAddChildId.contains(commonNodeParentRawType)) {
            commonNodeChildId = saturateChildId(commonNode.getData(Common.ChildId))
                    .toString();
        }


        if (m_CommandLineValues.Code2Seq) {
            stringBuilderMainPath.add(String.format("%s%s",
                    commonNode.getData(Common.PropertyKey).getType(true), commonNodeChildId));
        } else if (m_CommandLineValues.Code2Vec) {
            stringBuilderMainPath.add(String.format("%s%s%s%s", startSymbol,
                    commonNode.getData(Common.PropertyKey).getType(), commonNodeChildId, endSymbol));
        }
        if (m_CommandLineValues.GetJSON) {
            stringBuilderLongPath.add(String.format("%s%s",
                    commonNode.getData(Common.PropertyKey).getType(false), commonNodeChildId));
            idPath.add(commonNode.getData(Common.NodeId));
        }

        for (int i = targetStack.size() - commonPrefix - 1; i >= 0; i--) {
            Node currentNode = targetStack.get(i);
            String childId = Common.EmptyString;
            if (i == 0 || s_ParentTypeToAddChildId.contains(currentNode.getData(Common.PropertyKey).getRawType())) {
                childId = saturateChildId(currentNode.getData(Common.ChildId))
                        .toString();
            }

            if (m_CommandLineValues.Code2Seq) {
                stringBuilderMainPath.add(String.format("%s%s%s", shortSymbol,
                        currentNode.getData(Common.PropertyKey).getType(true), childId));
            } else if (m_CommandLineValues.Code2Vec) {
                stringBuilderMainPath.add(String.format("%s%s%s%s%s", downSymbol, startSymbol,
                        currentNode.getData(Common.PropertyKey).getType(), childId, endSymbol));
            }
            if (m_CommandLineValues.GetJSON) {
                stringBuilderLongPath.add(String.format("%s(%s%s)", downSymbol,
                        currentNode.getData(Common.PropertyKey).getType(false), childId));
                idPath.add(currentNode.getData(Common.NodeId));
            }
        }

        Paths paths = new Paths(stringBuilderMainPath.toString(),stringBuilderLongPath.toString(), idPath);

        return paths;
    }

    private Integer saturateChildId(int childId) {
        return Math.min(childId, m_CommandLineValues.MaxChildId);
    }
}
