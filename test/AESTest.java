import com.aes.AES;
import org.apache.commons.codec.binary.Hex;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;

public class AESTest {
    private final AES aes = new AES();

    @Test
    public void encryptCTRTest() {
        String key = "000102030405060708090a0b0c0d0e0f";
        String message = "8ea2b7ca516745bfeafc49904b496089";
        String iv = "11223344556677889911223344556677";

        List<Byte> encrypted = aes.encrypt(message, key, AES.Mode.CTR, iv);
    }

    //@Test
    public void padStringTest() {
        System.out.println(Hex.encodeHexString(aes.padString("1234567891234567")));
    }

    //    @Test
    public void grouperTest() {
        System.out.println(Arrays.toString(aes.grouper(aes.padString("12345678"), 3).toArray()));
    }

//    @Test
    public void hexStringToByteArrayTest() {
//        System.out.println(Arrays.toString(aes.hexStringToByteArray("11223344556677889911223344556677")));
        ByteBuffer bBuffer = ByteBuffer
                .allocate(4)
                .order(ByteOrder.LITTLE_ENDIAN);
        byte[] countBytes = bBuffer
                .putInt(0)
                .array();

        System.out.println(Arrays.toString(countBytes));
    }
}
