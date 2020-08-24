package Tokenizer;

import Tokenizer.Common.CommandLineValues;
import Tokenizer.Common.Common;
import Tokenizer.FeaturesEntities.MethodTokens;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

class ExtractTokensTask implements Callable<Void> {
    private final CommandLineValues m_CommandLineValues;
    private final Path filePath;

    public ExtractTokensTask(CommandLineValues commandLineValues, Path path) {
        this.m_CommandLineValues = commandLineValues;
        this.filePath = path;
    }

    @Override
    public Void call() {
        processFile();
        return null;
    }

    public void processFile() {
        ArrayList<MethodTokens> tokens;
        try {
            tokens = extractSingleFile();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        if (tokens == null) {
            return;
        }

        String toPrint = tokensToString(tokens);
        if (toPrint.length() > 0) {
            System.out.println(toPrint);
        }
    }

    public ArrayList<MethodTokens> extractSingleFile() throws IOException {
        String code = new String(Files.readAllBytes(this.filePath));
        TokenExtractor tokenExtractor = new TokenExtractor(this.m_CommandLineValues);
        ArrayList<MethodTokens> methodsTokens = tokenExtractor.extractTokens(code);


        if (this.m_CommandLineValues.PrintPaths) {
            methodsTokens.forEach(
                    methodTokens -> methodTokens.setMethodName(this.filePath + ":" + methodTokens.getMethodName()));
        }
        return methodsTokens;
    }

    public String tokensToString(ArrayList<MethodTokens> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            return Common.EmptyString;
        }

        List<String> methodsOutputs = new ArrayList<>();

        for (MethodTokens singleMethodTokens : tokens) {
            StringBuilder builder = new StringBuilder();

            String toPrint = singleMethodTokens.toString();
            if (m_CommandLineValues.PrettyPrint) {
                toPrint = toPrint.replace(" ", "\n\t");
            }
            builder.append(toPrint);

            methodsOutputs.add(builder.toString());
        }
        return Common.join(methodsOutputs, "\n");
    }
}
