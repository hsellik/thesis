package JavaExtractor.Common;

import org.apache.commons.cli.*;
import java.io.File;

/**
 * This class handles the programs arguments.
 */
public class CommandLineValues {
    //@Option(name = "--code2seq", required = false)
    public boolean Code2Seq = false;

    //@Option(name = "--code2vec", required = false)
    public boolean Code2Vec = false;

    //@Option(name = "--file", required = false)
    public File File = null;

    //@Option(name = "--dir", required = false, forbids = "--file")
    public String Dir = null;

    //@Option(name = "--max_path_length", required = true)
    public int MaxPathLength;

    //@Option(name = "--max_path_width", required = true)
    public int MaxPathWidth;

    //@Option(name = "--evaluate", required = false)
    public boolean Evaluate = false;

    //@Option(name = "--realistic_bug_ratio", required = false)
    public boolean RealisticBugRatio = false;

    //@Option(name = "--num_threads", required = false)
    public int NumThreads = 64;

    //@Option(name = "--min_code_len", required = false)
    public int MinCodeLength = 1;

    //@Option(name = "--max_code_len", required = false)
    public int MaxCodeLength = -1;

    //@Option(name = "--max_file_len", required = false)
    public int MaxFileLength = -1;

    //@Option(name = "--pretty_print", required = false)
    public boolean PrettyPrint = false;

    //@Option(name = "--max_child_id", required = false)
    public int MaxChildId = 3;

    //@Option(name = "--off_by_one", required = false, forbids = "--nullpointer")
    public boolean ProcessOffByOne = false;

    //@Option(name = "--nullpointer", required = false, forbids = "--off_by_one")
    public boolean ProcessNullpointer = false;

    //@Option(name = "--only_off_by_one_features", required = false, forbids = "--nullpointer")
    public boolean OnlyOffByOneFeatures = false;

    //@Option(name = "--get_json", required = false)
    public boolean GetJSON = false;


    public CommandLineValues(String... args) throws ParseException {
        Options options = new Options();

        Option code2seq = new Option(null, "code2seq", true, "Output features for code2seq model");
        options.addOption(code2seq);

        Option code2vec = new Option(null, "code2vec", true, "Output features for code2vec model");
        options.addOption(code2vec);

        Option file = new Option(null, "file", true, "Path of the file to be extracted");
        options.addOption(file);

        Option dir = new Option(null, "dir", true, "Path of the directory to be extracted");
        options.addOption(dir);

        Option maxPathLength = new Option(null, "max_path_length", true, "Max length of the paths");
        maxPathLength.setRequired(true);
        options.addOption(maxPathLength);

        Option maxPathWidth = new Option(null, "max_path_width", true, "Max width of the paths");
        maxPathWidth.setRequired(true);
        options.addOption(maxPathWidth);

        Option evaluate = new Option(null, "evaluate", true, "Evaluation mode");
        options.addOption(evaluate);

        Option realistic_bug_ratio = new Option(null, "realistic_bug_ratio", true, "Add less buggs to simulate a more realistic scenario");
        options.addOption(realistic_bug_ratio);

        Option numThreads = new Option(null, "num_threads", true, "Number of threads to use while extraction");
        options.addOption(numThreads);

        Option minCodeLength = new Option(null, "min_code_len", true, "Minimal code length to analyze");
        options.addOption(minCodeLength);

        Option maxCodeLength = new Option(null, "max_code_len", true, "Maximum code length to analyze");
        options.addOption(maxCodeLength);

        Option maxFileLength = new Option(null, "max_file_len", true, "Maximum file length to analyze");
        options.addOption(maxFileLength);

        Option prettyPrint = new Option(null, "pretty_print", true, "Enable pretty print");
        options.addOption(prettyPrint);

        Option maxChildId = new Option(null, "max_child_id", true, "");
        options.addOption(maxChildId);

        Option processOffByOne = new Option(null, "off_by_one", true, "Mutate to produce off-by-one errors");
        options.addOption(processOffByOne);

        Option processNullpointer = new Option(null, "nullpointer", true, "Mutate to remove nullpointer checks");
        options.addOption(processNullpointer);

        Option onlyOffByOneFeatures = new Option(null, "only_off_by_one_features", true, "Only outputs paths with nodes containing binary expressions");
        options.addOption(onlyOffByOneFeatures);

        Option getJSON = new Option(null, "get_json", true, "If true, return paths and AST in JSON format");
        options.addOption(getJSON);

        CommandLineParser parser = new GnuParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
            if (cmd.hasOption("code2seq")) Code2Seq = Boolean.parseBoolean(cmd.getOptionValue("code2seq"));
            if (cmd.hasOption("code2vec")) Code2Vec = Boolean.parseBoolean(cmd.getOptionValue("code2vec"));
            if (cmd.hasOption("file")) File = new File(cmd.getOptionValue("file"));
            if (cmd.hasOption("dir")) Dir = cmd.getOptionValue("dir");
            if (cmd.hasOption("max_path_length")) MaxPathLength = Integer.parseInt(cmd.getOptionValue("max_path_length"));
            if (cmd.hasOption("max_path_width")) MaxPathWidth = Integer.parseInt(cmd.getOptionValue("max_path_width"));
            if (cmd.hasOption("evaluate")) Evaluate = Boolean.parseBoolean(cmd.getOptionValue("evaluate"));
            if (cmd.hasOption("realistic_bug_ratio")) RealisticBugRatio = Boolean.parseBoolean(cmd.getOptionValue("realistic_bug_ratio"));
            if (cmd.hasOption("num_threads")) NumThreads = Integer.parseInt(cmd.getOptionValue("num_threads"));
            if (cmd.hasOption("min_code_len")) MinCodeLength = Integer.parseInt(cmd.getOptionValue("min_code_len"));
            if (cmd.hasOption("max_code_len")) MaxCodeLength = Integer.parseInt(cmd.getOptionValue("max_code_len"));
            if (cmd.hasOption("max_file_len")) MaxFileLength = Integer.parseInt(cmd.getOptionValue("max_file_len"));
            if (cmd.hasOption("pretty_print")) PrettyPrint = Boolean.parseBoolean(cmd.getOptionValue("pretty_print"));
            if (cmd.hasOption("max_child_id")) MaxChildId = Integer.parseInt(cmd.getOptionValue("max_child_id"));
            if (cmd.hasOption("off_by_one")) ProcessOffByOne = Boolean.parseBoolean(cmd.getOptionValue("off_by_one"));
            if (cmd.hasOption("nullpointer")) ProcessNullpointer = Boolean.parseBoolean(cmd.getOptionValue("nullpointer"));
            if (cmd.hasOption("only_off_by_one_features")) OnlyOffByOneFeatures = Boolean.parseBoolean(cmd.getOptionValue("only_off_by_one_features"));
            if (cmd.hasOption("get_json")) GetJSON = Boolean.parseBoolean(cmd.getOptionValue("get_json"));
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            throw (e);
        }
    }

    public CommandLineValues() {

    }
}