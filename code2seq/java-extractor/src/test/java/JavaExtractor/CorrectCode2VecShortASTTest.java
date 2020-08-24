package JavaExtractor;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CorrectCode2VecShortASTTest {
    @org.junit.jupiter.api.Test
    void main() {
        // Create a stream to hold the output
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos);
        PrintStream old = System.out;
        System.setOut(ps);

        String[] args = {"--code2vec", "true", "--max_path_length", "8", "--max_path_width", "2", "--off_by_one", "true", "--file", "src/test/resources/ShortJavaFileForAST.java"};
        App.main(args);

        System.out.flush();
        System.setOut(old);

        String result = baos.toString().split("\\n")[0].trim();
        assertEquals("bug public,713892080,METHOD_NAME public,950167328,void METHOD_NAME,-1051653051,void METHOD_NAME,-233248961,string METHOD_NAME,1438496582,a METHOD_NAME,2070746553,a METHOD_NAME,-46419787,a METHOD_NAME,1080690402,null METHOD_NAME,-1965427729,out METHOD_NAME,680169776,println METHOD_NAME,-33742705,asdfasd METHOD_NAME,-142652288,5 METHOD_NAME,-142652257,10 METHOD_NAME,-1965427729,out METHOD_NAME,680169776,println METHOD_NAME,-33742705,bee void,498829263,string void,1161853110,a void,-1425877431,a void,574196261,a void,157529170,null void,-2122519425,out void,601735840,println void,-1151045345,asdfasd void,671400432,5 void,671400463,10 void,-2122519425,out void,601735840,println void,-1151045345,bee string,-1874857689,a string,982592696,a string,1116924770,null string,966914304,5 string,966914335,10 a,250754796,a a,2056522089,a a,1515189910,null a,-927055308,5 a,-927055277,10 a,2110223822,a a,1913770683,null a,909870041,5 a,909870072,10 a,-311493177,null a,-1864988675,out a,1579600866,println a,-1080239715,asdfasd a,-741056001,5 a,-741055970,10 null,1907805228,system null,1353511473,out null,-285882834,println null,129626577,asdfasd null,468810291,5 null,468810322,10 null,60115363,println null,102404284,bee system,-1644303265,out system,-445258808,println system,-1060883721,asdfasd out,1037919652,println out,865870235,asdfasd println,-1909290285,asdfasd println,-491441099,5 println,-491441068,10 asdfasd,-406219536,5 asdfasd,-406219505,10 5,748445662,10 5,-1152511886,system 5,372612523,out 5,855222516,println 5,-333675445,bee 10,777718963,system 10,-1941192694,out 10,1655758069,println 10,360971562,bee system,-1644303265,out system,-445258808,println system,-1060883721,bee out,1037919652,println out,865870235,bee println,-1909290285,bee",
                result);
    }
}