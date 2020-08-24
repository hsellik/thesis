package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import org.apache.commons.cli.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

public class App {
    private static CommandLineValues s_CommandLineValues;
    public static String noBugString = "nobug";

    public static void main(String[] args) {
        try {
            if (args.length == 0) {
                s_CommandLineValues = new CommandLineValues("--max_path_length", "8", "--max_path_width", "2", "--off_by_one", "true", "--code2seq", "true", "--only_off_by_one_features", "false", "--get_json", "true", "--file",  "src/test/resources/ShortJavaFileForAST.java");
            } else {
                s_CommandLineValues = new CommandLineValues(args);
            }
        } catch (ParseException e) {
            e.printStackTrace();
            return;
        }

        if (s_CommandLineValues.File != null) {
            ExtractFeaturesTask extractFeaturesTask = new ExtractFeaturesTask(s_CommandLineValues,
                    s_CommandLineValues.File.toPath());
            extractFeaturesTask.processFile();
        } else if (s_CommandLineValues.Dir != null) {
            extractDir();
        }
    }

    private static void extractDir() {
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(s_CommandLineValues.NumThreads);
        LinkedList<ExtractFeaturesTask> tasks = new LinkedList<>();
        try {
            Files.walk(Paths.get(s_CommandLineValues.Dir)).filter(Files::isRegularFile)
                    .filter(p -> p.toString().toLowerCase().endsWith(".java")).forEach(f -> {
                ExtractFeaturesTask task = new ExtractFeaturesTask(s_CommandLineValues, f);
                tasks.add(task);
            });
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        List<Future<Void>> tasksResults = null;
        try {
            tasksResults = executor.invokeAll(tasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            executor.shutdown();
        }
        tasksResults.forEach(f -> {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        });
    }
}
