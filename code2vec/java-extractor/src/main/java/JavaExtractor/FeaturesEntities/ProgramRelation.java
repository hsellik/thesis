package JavaExtractor.FeaturesEntities;

import JavaExtractor.Common.CommandLineValues;

import java.util.List;

public class ProgramRelation {
    private final Property source;
    private final int sourceId;
    private final Property target;
    private final int targetId;
    private final String mainPath;
    private final String longPath;
    private final CommandLineValues m_commandLineValues;
    private final List<Integer> idPath;

    public ProgramRelation(CommandLineValues m_commandLineValues, Property sourceName, int sourceId, Property targetName, int targetId, String mainPath, String longPath, List<Integer> idPath) {
        this.m_commandLineValues = m_commandLineValues;
        this.source = sourceName;
        this.sourceId = sourceId;
        this.mainPath = mainPath;
        this.longPath = longPath;
        this.targetId = targetId;
        this.target = targetName;
        this.idPath = idPath;
    }

    public String toString() {
        if (m_commandLineValues.Code2Seq) {
            return String.format("%s,%s,%s", source.getName(), mainPath,
                    target.getName());
        } else if (m_commandLineValues.Code2Vec) {
            return String.format("%s,%s,%s", source.getName(), mainPath.hashCode(),
                    target.getName());
        } else {
            return "use either --code2seq or --code2vec flag!!!";
        }
    }

    public String getMainPath() {
        return mainPath;
    }

    public Property getSource() {
        return source;
    }

    public int getSourceId() {
        return sourceId;
    }

    public Property getTarget() {
        return target;
    }

    public int getTargetId() {
        return targetId;
    }

    public String getLongPath() {
        return longPath;
    }

    public List<Integer> getIdPath() {
        return idPath;
    }
}
