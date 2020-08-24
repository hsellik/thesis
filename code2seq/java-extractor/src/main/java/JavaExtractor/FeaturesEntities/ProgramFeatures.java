package JavaExtractor.FeaturesEntities;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import com.github.javaparser.ast.Node;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class ProgramFeatures {
    private String name;
    CommandLineValues m_CommandLineValues;
    private String originalOperator;
    private Node rootNode;

    private final ArrayList<ProgramRelation> features = new ArrayList<>();

    public ProgramFeatures(Node rootNode, String name, CommandLineValues m_CommandLineValues) {
        this.rootNode = rootNode;
        this.name = name;
        this.m_CommandLineValues = m_CommandLineValues;
    }

    @SuppressWarnings("StringBufferReplaceableByString")
    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        if (m_CommandLineValues.Evaluate) {
            stringBuilder.append(name).append(" ");
        } else {
            stringBuilder.append(name.contains("nobug") ? "nobug" : "bug").append(" ");
        }
        stringBuilder.append(features.stream().map(ProgramRelation::toString).collect(Collectors.joining(" ")));

        return stringBuilder.toString();
    }

    public String toJSON() {
        Gson gson = new GsonBuilder()
                .registerTypeAdapter(ProgramFeatures.class, new ProgramFeaturesAdapter())
                .setPrettyPrinting()
                .create();
        return gson.toJson(this, ProgramFeatures.class);
    }

    public void addFeature(Node source, String mainPath, String longPath, List<Integer> idPath, Node target) {
        Property sourceProperty = source.getData(Common.PropertyKey);
        int sourceId = source.getData(Common.NodeId);

        Property targetProperty = target.getData(Common.PropertyKey);
        int targetId = target.getData(Common.NodeId);

        ProgramRelation newRelation = new ProgramRelation(m_CommandLineValues, sourceProperty, sourceId, targetProperty, targetId, mainPath, longPath, idPath);
        features.add(newRelation);
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setOriginalOperator(String originalOperator) {
        this.originalOperator = originalOperator;
    }

    public boolean isEmpty() {
        return features.isEmpty();
    }

    public void removeNonOffByOnePaths() {
        String[] items = {"Gt", "Geq", "Ls", "Leq"};
        features.removeIf(feature -> Arrays.stream(items).parallel().noneMatch(feature.getMainPath()::contains));
    }

    public Node getRootNode() {
        return rootNode;
    }

    public ArrayList<ProgramRelation> getFeatures() {
        return features;
    }
}
