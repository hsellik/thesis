package JavaExtractor.Common;

import JavaExtractor.FeaturesEntities.ProgramFeatures;
import JavaExtractor.FeaturesEntities.Property;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.DataKey;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class Common {
    public static final DataKey<Property> PropertyKey = new DataKey<Property>() {
    };
    public static final DataKey<Integer> ChildId = new DataKey<Integer>() {
    };
    public static final DataKey<Integer> NodeId = new DataKey<Integer>() {};
    public static final String EmptyString = "";

    public static final String MethodDeclaration = "MethodDeclaration";
    public static final String SimpleName = "SimpleName";
    public static final String NameExpr = "NameExpr";
    public static final String BlankWord = "BLANK";

    public static final int c_MaxLabelLength = 50;
    public static final String methodName = "METHOD_NAME";
    public static final String internalSeparator = "|";

    public static String normalizeName(String original, String defaultString) {
        original = original.toLowerCase().replaceAll("\\\\n", "") // escaped new
                // lines
                .replaceAll("//s+", "") // whitespaces
                .replaceAll("[\"',]", "") // quotes, apostrophies, commas
                .replaceAll("\\P{Print}", ""); // unicode weird characters
        String stripped = original.replaceAll("[^A-Za-z]", "");
        if (stripped.length() == 0) {
            String carefulStripped = original.replaceAll(" ", "_");
            if (carefulStripped.length() == 0) {
                return defaultString;
            } else {
                return carefulStripped;
            }
        } else {
            return stripped;
        }
    }

    public static boolean isMethod(Node node, String type) {
        if (!node.getParentNode().get().containsData(Common.PropertyKey)) {
            return false;
        }
        Property parentProperty = node.getParentNode().get().getData(Common.PropertyKey);

        String parentType = parentProperty.getType();
        return Common.SimpleName.equals(type) && Common.MethodDeclaration.equals(parentType);
    }

    public static ArrayList<String> splitToSubtokens(String str1) {
        String str2 = str1.replace("|", " ");
        String str3 = str2.trim();
        return Stream.of(str3.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+"))
                .filter(s -> s.length() > 0).map(s -> Common.normalizeName(s, Common.EmptyString))
                .filter(s -> s.length() > 0).collect(Collectors.toCollection(ArrayList::new));
    }

    public static String join(Iterable<?> iterable, String separator) {
        return iterable == null ? null : join(iterable.iterator(), separator);
    }

    public static String join(Iterator<?> iterator, String separator) {
        if (iterator == null) {
            return null;
        } else if (!iterator.hasNext()) {
            return "";
        } else {
            Object first = iterator.next();
            if (!iterator.hasNext()) {
                return first == null ? "" : first.toString();
            } else {
                StringBuilder buf = new StringBuilder(256);
                if (first != null) {
                    buf.append(first);
                }

                while(iterator.hasNext()) {
                    if (separator != null) {
                        buf.append(separator);
                    }

                    Object obj = iterator.next();
                    if (obj != null) {
                        buf.append(obj);
                    }
                }

                return buf.toString();
            }
        }
    }

    public static String featuresToJSON(ArrayList<ProgramFeatures> features) {
        return features != null && !features.isEmpty() ? features.get(0).toJSON() : "";
    }
}
