package Tokenizer.Common;

import org.apache.commons.cli.*;

import java.io.File;

/**
 * This class handles the program arguments.
 */
public class CommandLineValues {
    //@Option(name = "--file", required = false)
    public File File = null;

    //@Option(name = "--dir", required = false, forbids = "--file")
    public String Dir = null;

    //@Option(name = "--sequence_length", required = false)
    public int SequenceLength = 5;

    //@Option(name = "--print_paths", required = false)
    public boolean PrintPaths = true;

    //@Option(name = "--num_threads", required = false)
    public int NumThreads = 32;

    //@Option(name = "--pretty_print", required = false)
    public boolean PrettyPrint = false;


    public CommandLineValues(String... args) throws ParseException {
        Options options = new Options();

        Option file = new Option(null, "file", true, "Path of the file to be extracted");
        options.addOption(file);

        Option dir = new Option(null, "dir", true, "Path of the directory to be extracted");
        options.addOption(dir);

        Option sequenceLength = new Option(null, "sequence_length", true, "Long token sequences will be cut to this value");
        options.addOption(sequenceLength);

        Option printPaths = new Option(null, "print_paths", true, "Print paths in addition to method names");
        options.addOption(printPaths);

        Option numThreads = new Option(null, "num_threads", true, "Number of threads to use while extraction");
        options.addOption(numThreads);

        Option prettyPrint = new Option(null, "pretty_print", true, "Enable pretty print");
        options.addOption(prettyPrint);

        CommandLineParser parser = new GnuParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
            if (cmd.hasOption("file")) File = new File(cmd.getOptionValue("file"));
            if (cmd.hasOption("dir")) Dir = cmd.getOptionValue("dir");
            if (cmd.hasOption("sequence_length"))
                SequenceLength = Integer.parseInt(cmd.getOptionValue("sequence_length"));
            if (cmd.hasOption("print_paths")) PrintPaths = Boolean.parseBoolean(cmd.getOptionValue("print_paths"));
            if (cmd.hasOption("num_threads")) NumThreads = Integer.parseInt(cmd.getOptionValue("num_threads"));
            if (cmd.hasOption("pretty_print")) PrettyPrint = Boolean.parseBoolean(cmd.getOptionValue("pretty_print"));
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            throw (e);
        }
    }

    public CommandLineValues() {

    }
}