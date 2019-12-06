import com.aes.AES;
import org.apache.commons.codec.binary.Hex;
import org.junit.Test;
import java.util.Arrays;

public class AESTest {
    private final AES aes = new AES();

    //@Test
    public void padStringTest() {
        System.out.println(Hex.encodeHexString(aes.padString("1234567891234567")));
    }

    @Test
    public void grouperTest() {
        System.out.println(Arrays.toString(aes.grouper(aes.padString("12345678"), 3).toArray()));
    }
}
