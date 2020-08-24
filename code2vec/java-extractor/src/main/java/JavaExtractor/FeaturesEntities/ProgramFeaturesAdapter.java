package JavaExtractor.FeaturesEntities;

import JavaExtractor.Common.Common;
import com.github.javaparser.ast.Node;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;

import java.io.IOException;
import java.util.ArrayList;

public class ProgramFeaturesAdapter extends TypeAdapter<ProgramFeatures> {

    Gson gson = new GsonBuilder().create();

    public void write(JsonWriter writer, ProgramFeatures programFeatures) throws IOException {
        if (programFeatures == null) {
            writer.nullValue();
            return;
        }
        writer.beginObject();

        writeModelInput(writer, programFeatures.toString());
        writeAllPaths(writer, programFeatures.getFeatures());
        writeAST(writer, programFeatures.getRootNode());

        writer.endObject();
    }

    private void writeModelInput(JsonWriter writer, String input) throws IOException {
        if (input == null || input.isEmpty()) {
            return;
        }
        writer.name("modelInput").value(input);
    }

    private void writeAllPaths(JsonWriter writer, ArrayList<ProgramRelation> features) throws IOException {
        if (features == null) {
            return;
        }
        writer.name("paths");
        writer.beginArray();

        for (ProgramRelation feature: features) {
            writePath(writer, feature);
        }

        writer.endArray();
    }

    private void writePath(JsonWriter writer, ProgramRelation feature) throws IOException {
        if (feature == null) {
            return;
        }
        writer.beginObject();

        writer.name("source").value(feature.getSource().getName());
        writer.name("sourceId").value(feature.getSourceId());
        writer.name("target").value(feature.getTarget().getName());
        writer.name("targetId").value(feature.getTargetId());
        writer.name("mainPath").value(feature.getMainPath());
        writer.name("longPath").value(feature.getLongPath());
        writer.name("idPath").value(feature.getIdPath().toString());

        writer.endObject();
    }

    private void writeAST(JsonWriter writer, Node rootNode) throws IOException {
        if (rootNode == null) {
            return;
        }
        writer.name("ast");
        writeNodeAndChildren(writer, rootNode);
    }

    private void writeNodeAndChildren(JsonWriter writer, Node node) throws IOException {
        writer.beginObject();

        // write node
        writer.name("id").value(node.getData(Common.NodeId));
        if (node.getRange().isPresent()) {
            writer.name("range").jsonValue(this.gson.toJson(node.getRange().get()));
        }

        // write children
        if (node.getChildNodes().size() == 0) {
            writer.name("name").value(node.getData(Common.PropertyKey).getName());
            writer.endObject();
            return;
        } else {
            writer.name("name").value(node.getData(Common.PropertyKey).getType());
        }
        writer.name("children");
        writer.beginArray();

        for (Node child : node.getChildNodes()) {
            writeNodeAndChildren(writer, child);
        }
        writer.endArray();

        writer.endObject();

    }

    @Override
    public ProgramFeatures read(JsonReader in) throws IOException {
        return null;
    }
}