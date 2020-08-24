package JavaExtractor.Mutator;

import JavaExtractor.FeaturesEntities.ExtractedMethod;
import com.github.javaparser.ast.body.MethodDeclaration;

import java.util.List;

public interface MethodMutator {

    List<ExtractedMethod> processMethod(MethodDeclaration method);

}
