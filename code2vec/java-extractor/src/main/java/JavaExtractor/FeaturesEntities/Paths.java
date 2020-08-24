package JavaExtractor.FeaturesEntities;

import java.util.List;

public class Paths {

    private final String mainPath;
    private final String longPath;
    private final List<Integer> idPath;

    public Paths(String mainPath, String longPath, List<Integer> idPath) {
        this.mainPath = mainPath;
        this.longPath = longPath;
        this.idPath = idPath;
    }

    public String getMainPath() {
        return mainPath;
    }

    public String getLongPath() {
        return longPath;
    }

    public List<Integer> getIdPath() {
        return idPath;
    }
}
