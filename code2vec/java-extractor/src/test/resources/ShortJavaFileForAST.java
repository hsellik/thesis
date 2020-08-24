package JavaExtractor;

import java.io.IOException;

public class ExtractFeaturesTask implements Callable<Void> {

    public void processCode() {
        String a = "a";
        if (a != null) {
            System.out.println("asdfasd");
        } if (5 < 10) {
            System.out.println("bee");
        }
    }
}
