package JavaExtractor;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.assertTrue;

// Test to make sure basic output paths do not change with small changes / added features
class CorrectCode2SeqShortASTTest {
    @org.junit.jupiter.api.Test
    void main() {
        // Create a stream to hold the output
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);

        String[] args = {"--code2seq", "true", "--max_path_length", "8", "--max_path_width", "2", "--file", "src/test/resources/ShortJavaFileForAST.java", "--off_by_one", "true"};
        App.main(args);

        System.out.flush();
        System.setOut(old);
        String result = baos.toString();

        assertTrue(result.contains("bug public,Mdfr0|Mth|SmplNm1,METHOD_NAME public,Mdfr0|Mth|Void2,void METHOD_NAME,SmplNm1|Mth|Void2,void METHOD_NAME,SmplNm1|Mth|Bk|Ex|VDE|VD|Cls|SmplNm0,string METHOD_NAME,SmplNm1|Mth|Bk|Ex|VDE|VD|SmplNm1,a METHOD_NAME,SmplNm1|Mth|Bk|Ex|VDE|VD|StrEx2,a METHOD_NAME,SmplNm1|Mth|Bk|If|Neq|Nm|SmplNm0,a METHOD_NAME,SmplNm1|Mth|Bk|If|Neq|Null1,null METHOD_NAME,SmplNm1|Mth|Bk|If|Bk|Ex|Cal0|Fld0|SmplNm1,out METHOD_NAME,SmplNm1|Mth|Bk|If|Bk|Ex|Cal0|SmplNm1,println METHOD_NAME,SmplNm1|Mth|Bk|If|Bk|Ex|Cal0|StrEx2,asdfasd METHOD_NAME,SmplNm1|Mth|Bk|If|Leq|IntEx0,5 METHOD_NAME,SmplNm1|Mth|Bk|If|Leq|IntEx1,10 METHOD_NAME,SmplNm1|Mth|Bk|If|Bk|Ex|Cal0|Fld0|SmplNm1,out METHOD_NAME,SmplNm1|Mth|Bk|If|Bk|Ex|Cal0|SmplNm1,println METHOD_NAME,SmplNm1|Mth|Bk|If|Bk|Ex|Cal0|StrEx2,bee void,Void2|Mth|Bk|Ex|VDE|VD|Cls|SmplNm0,string void,Void2|Mth|Bk|Ex|VDE|VD|SmplNm1,a void,Void2|Mth|Bk|Ex|VDE|VD|StrEx2,a void,Void2|Mth|Bk|If|Neq|Nm|SmplNm0,a void,Void2|Mth|Bk|If|Neq|Null1,null void,Void2|Mth|Bk|If|Bk|Ex|Cal0|Fld0|SmplNm1,out void,Void2|Mth|Bk|If|Bk|Ex|Cal0|SmplNm1,println void,Void2|Mth|Bk|If|Bk|Ex|Cal0|StrEx2,asdfasd void,Void2|Mth|Bk|If|Leq|IntEx0,5 void,Void2|Mth|Bk|If|Leq|IntEx1,10 void,Void2|Mth|Bk|If|Bk|Ex|Cal0|Fld0|SmplNm1,out void,Void2|Mth|Bk|If|Bk|Ex|Cal0|SmplNm1,println void,Void2|Mth|Bk|If|Bk|Ex|Cal0|StrEx2,bee string,SmplNm0|Cls|VD|SmplNm1,a string,SmplNm0|Cls|VD|StrEx2,a string,SmplNm0|Cls|VD|VDE|Ex|Bk|If|Neq|Null1,null string,SmplNm0|Cls|VD|VDE|Ex|Bk|If|Leq|IntEx0,5 string,SmplNm0|Cls|VD|VDE|Ex|Bk|If|Leq|IntEx1,10 a,SmplNm1|VD|StrEx2,a a,SmplNm1|VD|VDE|Ex|Bk|If|Neq|Nm|SmplNm0,a a,SmplNm1|VD|VDE|Ex|Bk|If|Neq|Null1,null a,SmplNm1|VD|VDE|Ex|Bk|If|Leq|IntEx0,5 a,SmplNm1|VD|VDE|Ex|Bk|If|Leq|IntEx1,10 a,StrEx2|VD|VDE|Ex|Bk|If|Neq|Nm|SmplNm0,a a,StrEx2|VD|VDE|Ex|Bk|If|Neq|Null1,null a,StrEx2|VD|VDE|Ex|Bk|If|Leq|IntEx0,5 a,StrEx2|VD|VDE|Ex|Bk|If|Leq|IntEx1,10 a,SmplNm0|Nm|Neq|Null1,null a,SmplNm0|Nm|Neq|If|Bk|Ex|Cal0|Fld0|SmplNm1,out a,SmplNm0|Nm|Neq|If|Bk|Ex|Cal0|SmplNm1,println a,SmplNm0|Nm|Neq|If|Bk|Ex|Cal0|StrEx2,asdfasd a,SmplNm0|Nm|Neq|If|Bk|If|Leq|IntEx0,5 a,SmplNm0|Nm|Neq|If|Bk|If|Leq|IntEx1,10 null,Null1|Neq|If|Bk|Ex|Cal0|Fld0|Nm|SmplNm0,system null,Null1|Neq|If|Bk|Ex|Cal0|Fld0|SmplNm1,out null,Null1|Neq|If|Bk|Ex|Cal0|SmplNm1,println null,Null1|Neq|If|Bk|Ex|Cal0|StrEx2,asdfasd null,Null1|Neq|If|Bk|If|Leq|IntEx0,5 null,Null1|Neq|If|Bk|If|Leq|IntEx1,10 null,Null1|Neq|If|Bk|If|Bk|Ex|Cal0|SmplNm1,println null,Null1|Neq|If|Bk|If|Bk|Ex|Cal0|StrEx2,bee system,SmplNm0|Nm0|Fld0|SmplNm1,out system,SmplNm0|Nm0|Fld0|Cal|SmplNm1,println system,SmplNm0|Nm0|Fld0|Cal|StrEx2,asdfasd out,SmplNm1|Fld0|Cal|SmplNm1,println out,SmplNm1|Fld0|Cal|StrEx2,asdfasd println,SmplNm1|Cal|StrEx2,asdfasd println,SmplNm1|Cal|Ex|Bk|If|Bk|If|Leq|IntEx0,5 println,SmplNm1|Cal|Ex|Bk|If|Bk|If|Leq|IntEx1,10 asdfasd,StrEx2|Cal|Ex|Bk|If|Bk|If|Leq|IntEx0,5 asdfasd,StrEx2|Cal|Ex|Bk|If|Bk|If|Leq|IntEx1,10 5,IntEx0|Leq|IntEx1,10 5,IntEx0|Leq|If|Bk|Ex|Cal0|Fld0|Nm|SmplNm0,system 5,IntEx0|Leq|If|Bk|Ex|Cal0|Fld0|SmplNm1,out 5,IntEx0|Leq|If|Bk|Ex|Cal0|SmplNm1,println 5,IntEx0|Leq|If|Bk|Ex|Cal0|StrEx2,bee 10,IntEx1|Leq|If|Bk|Ex|Cal0|Fld0|Nm|SmplNm0,system 10,IntEx1|Leq|If|Bk|Ex|Cal0|Fld0|SmplNm1,out 10,IntEx1|Leq|If|Bk|Ex|Cal0|SmplNm1,println 10,IntEx1|Leq|If|Bk|Ex|Cal0|StrEx2,bee system,SmplNm0|Nm0|Fld0|SmplNm1,out system,SmplNm0|Nm0|Fld0|Cal|SmplNm1,println system,SmplNm0|Nm0|Fld0|Cal|StrEx2,bee out,SmplNm1|Fld0|Cal|SmplNm1,println out,SmplNm1|Fld0|Cal|StrEx2,bee println,SmplNm1|Cal|StrEx2,bee"));
    }
}