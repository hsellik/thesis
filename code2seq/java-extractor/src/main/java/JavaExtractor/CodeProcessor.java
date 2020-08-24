package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;

public class CodeProcessor {
    private final CommandLineValues m_CommandLineValues;

    public CodeProcessor(CommandLineValues commandLineValues) {
        this.m_CommandLineValues = commandLineValues;
    }

    public String processCode(String code) {
        FeatureExtractor featureExtractor = new FeatureExtractor(this.m_CommandLineValues);
        return Common.featuresToJSON(featureExtractor.extractFeatures(code));
    }
}
