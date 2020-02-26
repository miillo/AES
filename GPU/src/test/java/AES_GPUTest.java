import com.aes.AES_GPU;
import com.aes.gpu.JCudaTest;
import com.google.common.primitives.Bytes;
import org.apache.commons.codec.binary.Hex;
import org.junit.Test;

import javax.crypto.*;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class AES_GPUTest {

    @Test
    public void encryptCTRTest() {
        /*JCUDA MOCK DATA*/
        String ptxFilePath = "D:\\Coding\\Java\\AES\\GPU\\src\\main\\resources\\galoisMul.ptx";
        String funcName = "galoisMul";
        /**/

        /*TEST DATA*/
        //key length: 16B / 24B / 32B
        String key = "000102030405060708090a0b0c0d0e0f0a0b0c0d0e0f0a0b";
        //Any length
        String message = "8ea2b7ca516745bfeafc49904b496089";
        //IV for CTR may be any values LESS THAN 16B
        String iv = "11223344556677889911223344556677";
        /**/

        AES_GPU aes_gpu = new AES_GPU(ptxFilePath, funcName);

        //our implementation
        List<Byte> encrypted = aes_gpu.encrypt(message, key, true, iv);
        String encryptedStr = Hex.encodeHexString(Bytes.toArray(encrypted));

        //built in
        IvParameterSpec ivspec = new IvParameterSpec(aes_gpu.hexStringToByteArray(iv));
        SecretKey secretKey = new SecretKeySpec(aes_gpu.hexStringToByteArray(key), 0, aes_gpu.hexStringToByteArray(key).length, "AES");
        Cipher cipher;
        String cipheredMessageStr = null;
        try {
            cipher = Cipher.getInstance("AES/CTR/NoPadding");
            cipher.init(Cipher.ENCRYPT_MODE, secretKey, ivspec);
            byte[] cipheredMessage = cipher.doFinal(aes_gpu.hexStringToByteArray(message));
            cipheredMessageStr = Hex.encodeHexString(cipheredMessage);
        } catch (NoSuchAlgorithmException | NoSuchPaddingException | InvalidAlgorithmParameterException | InvalidKeyException | BadPaddingException | IllegalBlockSizeException e) {
            e.printStackTrace();
        }
        System.out.println(encryptedStr);
        System.out.println(cipheredMessageStr);
        assertEquals(cipheredMessageStr, encryptedStr);
    }

    @Test
    public void executeGPUTestsTest() {
        JCudaTest.printGPUInfo();
    }
}
