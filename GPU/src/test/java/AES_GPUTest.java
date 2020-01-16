import com.aes.AES;
import com.aes.AES_GPU;
import com.aes.gpu.AES_JCuda;
import com.google.common.primitives.Bytes;
import org.apache.commons.codec.binary.Hex;
import org.junit.Test;

import javax.crypto.*;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AES_GPUTest {
    private final AES_GPU aes = new AES_GPU();
    private final AES_JCuda aes_jCuda = new AES_JCuda();

    @Test
    public void executeGPUTestsTest() {
        aes_jCuda.executeGPUTestsTest();
    }

    @Test
    public void encryptCTRTest() {
        //key length: 16B / 24B / 32B
        String key = "000102030405060708090a0b0c0d0e0f0a0b0c0d0e0f0a0b";
        //Any length
        String message = "8ea2b7ca516745bfeafc49904b496089";
        //IV for CTR may be any values LESS THAN 16B
        String iv = "11223344556677889911223344556677";

        //our implementation
        List<Byte> encrypted = aes.encrypt(message, key, true, iv);
        String encryptedStr = Hex.encodeHexString(Bytes.toArray(encrypted));

        //built in
        IvParameterSpec ivspec = new IvParameterSpec(aes.hexStringToByteArray(iv));
        SecretKey secretKey = new SecretKeySpec(aes.hexStringToByteArray(key), 0, aes.hexStringToByteArray(key).length, "AES");
        Cipher cipher;
        String cipheredMessageStr = null;
        try {
            cipher = Cipher.getInstance("AES/CTR/NoPadding");
            cipher.init(Cipher.ENCRYPT_MODE, secretKey, ivspec);
            byte[] cipheredMessage = cipher.doFinal(aes.hexStringToByteArray(message));
            cipheredMessageStr = Hex.encodeHexString(cipheredMessage);
        } catch (NoSuchAlgorithmException | NoSuchPaddingException | InvalidAlgorithmParameterException | InvalidKeyException | BadPaddingException | IllegalBlockSizeException e) {
            e.printStackTrace();
        }
        System.out.println(encryptedStr);
        System.out.println(cipheredMessageStr);
        assertEquals(cipheredMessageStr, encryptedStr);
    }

    @Test
    public void galoisMulTest() {
        System.out.println(aes.galoisMul((byte)30, (byte)90)  & 0xFF);
    }
}
