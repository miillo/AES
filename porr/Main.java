package com.aes;
import com.google.common.primitives.Bytes;
import org.apache.commons.codec.binary.Hex;
import org.apache.commons.lang3.ArrayUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;
public class Main {
	
private Map<Integer, Integer> rounds = new HashMap<Integer, Integer>(){{
        put(128,10);
        put(192, 12);
        put(256, 14);

    }};
	//max key expansions lengths
    private Map<Integer, Integer> keyExpansionMaxSizes = new HashMap<Integer, Integer>(){
        {
            put(16, 176);
            put(24, 208);
            put(32, 240);
        }};


    //S-Box table for Byte Substitution layer -> taken from official documentation
    //previously was 2d int matrix
    private int[] sBox = {
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
    };

    //todo: add description from docs
    private int[] rcon = {
            0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a,
            0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39,
            0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a,
            0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8,
            0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef,
            0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc,
            0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b,
            0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3,
            0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94,
            0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20,
            0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97, 0x35,
            0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd, 0x61, 0xc2, 0x9f,
            0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d, 0x01, 0x02, 0x04,
            0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36, 0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63,
            0xc6, 0x97, 0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39, 0x72, 0xe4, 0xd3, 0xbd,
            0x61, 0xc2, 0x9f, 0x25, 0x4a, 0x94, 0x33, 0x66, 0xcc, 0x83, 0x1d, 0x3a, 0x74, 0xe8, 0xcb, 0x8d
    };

    public static void main(String[] args) {
		Main main = new Main();
		//System.out.println(main.bytesToHex("Any String you want".getBytes()));
		System.out.println(main.keyExpansion(main.hexStringToByteArray("000102030405060708090a0b0c0d0e0f")));

    }
	    /**
     * Converts bytes array to hex string
     *
     * @param bytes bytes array
     * @return hex string
     */
	public String bytesToHex(byte[] bytes) {
        char[] HEX_ARRAY = "0123456789ABCDEF".toCharArray();
		System.out.println(bytes.length);
        char[] hexChars = new char[bytes.length * 4];
		char[][] hexCharList = new char[2][bytes.length * 2];
		// omp parallel threadNum(4)
		{
		// omp for 
        for (int j = 0; j < bytes.length; j++) {
			System.out.println(bytes[j] + " " + OMP4J_THREAD_NUM + " " + OMP4J_NUM_THREADS);
            int v = bytes[j] & 0xFF;
            hexChars[j * 2] = HEX_ARRAY[v >>> 4];
			hexChars[j * 2 + 1] = HEX_ARRAY[v & 0x0F];
        }
		
		}
        return new String(hexChars);
    }
	
	 /**
     * Encrypts message using AES
     *
     * @param message message for encryption
     * @param key     key
     * @param iv      initialization vector[CBC specific]
     */
    public List<Byte> encrypt(String message, String key, boolean isHex, String iv) {
        byte[] paddedMsg;
        if (isHex) {
            paddedMsg = hexStringToByteArray(message);
        } else {
            paddedMsg = padString(message); //todo: should be tested
        }

        List<List<Byte>> blocks = grouper(paddedMsg, 16);
        List<Byte> ret = new ArrayList<>();

		byte[] ivBytes = hexStringToByteArray(iv);
		//strange cuz' this loop runs always once, but it was in friend's code
		for (int i = 0; i < blocks.size(); i++) {
			byte[] countBytes = convertIntToByteArray(i);
			byte[] aesIn = xorArraysWithSameLength(ivBytes, countBytes);
			byte[] countCiphered = encryptBlock(key, aesIn, isHex);
			byte[] cipherText = xorArraysWithSameLength(Bytes.toArray(blocks.get(i)), countCiphered);
			ret.addAll(Arrays.asList(ArrayUtils.toObject(cipherText)));
		}

        return ret;
    }

    /**
     * Length of the plaintext should be an multiple of the block size of the cipher[16B - AES]
     * Function pads '1' and rest bits with zeros
     *
     * @param message message for padding
     * @return message as byte array in HEX format
     */
    public byte[] padString(String message) {
        ByteArrayOutputStream bStream = new ByteArrayOutputStream();
        byte[] msgBits = message.getBytes(StandardCharsets.UTF_8);
        try {
            bStream.write(msgBits);
            bStream.write(128);
            //count how many 0s is needed for padding
            int count = 16 - bStream.toByteArray().length % 16;
            for (int i = 0; i < count; i++)
                bStream.write(0);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bStream.toByteArray();
    }

    /**
     * Collects data into fixed-length chunks or blocks
     * e.g. grouper([1,2,3,4,5,6,7], 3) --> [[1,2,3], [4,5,6] [7,null,null]]"
     *
     * @param paddedMsg padded message
     * @param size      size of a block
     * @return grouped iterable i.e. 'list of lists'
     */
    public List<List<Byte>> grouper(byte[] paddedMsg, int size) {
        List<List<Byte>> resultList = new ArrayList<>();

        int helper = 0;
        List<Byte> tmp = new ArrayList<>();
		
        for (byte b : paddedMsg) {
            if (helper < size) {
                tmp.add(b);
                helper++;
            } else {
                helper = 1;
                resultList.add(tmp);
                tmp = new ArrayList<>();
                tmp.add(b);
            }
        }

        //filling 'free' spaces up to size with nulls
        if (tmp.size() < size) {
            for (int i = tmp.size(); i < size; i++)
                tmp.add(null);
        }
        resultList.add(tmp);
        return resultList;
    }

    /**
     * Converts hex string to byte array
     *
     * @param s hex string
     * @return byte array
     */
    public byte[] hexStringToByteArray(String s) {
        int len = s.length();
        byte[] data = new byte[len / 2];
        for (int i = 0; i < len; i += 2) {
            data[i / 2] = (byte) ((Character.digit(s.charAt(i), 16) << 4)
                    + Character.digit(s.charAt(i + 1), 16));
        }

        return data;
    }

    /**
     * Converts integer to byte array
     *
     * @param i integer for conversion
     * @return byte array
     */
    public byte[] convertIntToByteArray(int i) {
        ByteBuffer bBuffer = ByteBuffer
                .allocate(16)
                .order(ByteOrder.LITTLE_ENDIAN);
        return bBuffer
                .putInt(i)
                .array();
    }

    /**
     * Performs XOR operation on arrays with same length
     *
     * @param arr1 array 1
     * @param arr2 array 2
     * @return new array with XOR'ed values
     */
    public byte[] xorArraysWithSameLength(byte[] arr1, byte[] arr2) {
        byte[] result = new byte[arr1.length];
        for (int i = 0; i < result.length; i++)
            result[i] = (byte) (arr1[i] ^ arr2[i]);
        return result;
    }

    /**
     * Encrypts block with AES algorithm
     *
     * @param key   key
     * @param text  text for encryption
     * @param isHex whether text is in HEX format
     * @return encrypted block of size 4B
     */
    public byte[] encryptBlock(String key, byte[] text, boolean isHex) {
        List<List<Byte>> state = grouper(text, 4);
        byte[] keyBytes;
        if (isHex) {
            keyBytes = hexStringToByteArray(key);
        } else {
            keyBytes = new byte[16]; //todo: how can we do it?
        }
        int rounds = this.rounds.get(keyBytes.length * 8);
        byte[] expandedKey = keyExpansion(keyBytes);

        byte[] fstKey = new byte[16];
        System.arraycopy(expandedKey, 0, fstKey, 0, 16);

        List<List<Byte>> groupedFstKey = grouper(fstKey, 4);
        state = addRoundKey(state, groupedFstKey);

        for (int i = 1; i < rounds; i++) {
            state = subBytes(state);
            state = shiftRows(state);
            state = mixColumns(state);
            byte[] roundKey = createRoundKey(expandedKey, i);
            List<List<Byte>> groupedRoundKey = grouper(roundKey, 4);
            state = addRoundKey(state, groupedRoundKey);
        }

        state = subBytes(state);
        state = shiftRows(state);

        byte[] tmpExpandedKey = Arrays.copyOfRange(expandedKey, expandedKey.length - 16, expandedKey.length);
        List<List<Byte>> groupedTmpExpandedKey = grouper(tmpExpandedKey, 4);
        state = addRoundKey(state, groupedTmpExpandedKey);

        //merging state structure
        List<Byte> mergedState = new ArrayList<>();
        for (List<Byte> l : state)
            mergedState.addAll(l);

        return Bytes.toArray(mergedState);
    }


    /**
     * Key expansion operation - https://www.samiam.org/key-schedule.html
     *
     * @param cipherKey cipher key
     * @return expanded key
     */
    public byte[] keyExpansion(byte[] cipherKey) {
		System.out.println(cipherKey);
		System.out.println("1");
        int cipherKeySize = cipherKey.length;
		System.out.println(cipherKey.length);
        List<Integer> expandedKey = new ArrayList<>(); //container for expanded key | ? not sure if byte[] or list ? int or byte?
        int currentSize = 0;
        int rconIteration = 1;
        int[] temp = {0, 0, 0, 0}; //temporary list to store 4B at a time
		System.out.println("1");
        //copy the first cipher_key bytes of the cipher key to the expanded key
        for (byte b : cipherKey) {
            expandedKey.add((int) b);
        }
        currentSize += cipherKeySize;

        for (;expandedKey.size() < keyExpansionMaxSizes.get(cipherKeySize);) {
//        while (expandedKey.size() < keyExpansionMaxSizes.get(cipherKeySize)) {
            //assign previous 4 bytes to the temporary storage t
            for (int i = 0; i < 4; i++) {
                temp[i] = expandedKey.get((currentSize - 4) + i);
            }
		System.out.println("while");
            //every 32 bytes apply the core schedule to t
            if (cipherKeySize == 32 && currentSize % 32 == 0) {
						System.out.println("1");
                temp = keyScheduleCore(temp, rconIteration);
						System.out.println("2");
                rconIteration += 1;
            }

            if (cipherKeySize == 16 && currentSize % 16 == 0) {
               System.out.println("1");
                temp = keyScheduleCore(temp, rconIteration);
						System.out.println("2");
                rconIteration += 1;
            }

            if (cipherKeySize == 24 && currentSize % 24 == 0) {
                temp = keyScheduleCore(temp, rconIteration);
                rconIteration += 1;
            }

            //since we're using a 256-bit key -> add an extra sbox transform
            if (cipherKeySize == 32 && currentSize % cipherKeySize == 16) {
                for (int i = 0; i < 4; i++) {
                    temp[i] = this.sBox[temp[i]];
                }
            }

            //XOR t with the 4-byte block [16,24,32] bytes before the end of the current expanded key.
            //These 4 bytes become the next bytes in the expanded key
            for (int i = 0; i < 4; i++) {
                expandedKey.add((expandedKey.get(currentSize - cipherKeySize)) ^ (temp[i]));
                currentSize += 1;
            }
        }
		System.out.println("po while");
        Byte[] convertedExKey = expandedKey
                .stream()
                .map(Integer::byteValue)
                .toArray(Byte[]::new);

        return ArrayUtils.toPrimitive(convertedExKey);
    }

    /**
     * Generates key core - > https://www.samiam.org/key-schedule.html
     *
     * @param temp          temporary array for storing 4B at a time i.e. word
     * @param rconIteration iteration number
     * @return generated key core
     */
    public int[] keyScheduleCore(int[] temp, int rconIteration) {
        //rotate word 1 byte to the left
        temp = rotateArray(temp, 1);
        int[] newTemp = {0, 0, 0, 0};
        //apply sbox substitution on all bytes of word
        for (int i = 0; i < temp.length; i++) {
            newTemp[i] = this.sBox[temp[i]];
        }
        //XOR the output of the rcon[i] transformation with the first part of the word
        newTemp[0] = newTemp[0] ^ this.rcon[rconIteration];
        return newTemp;
    }

    /**
     * Rotates array by n elements
     *
     * @param array array
     * @param n     no. of rotation elements
     * @return rotated array
     */
    public int[] rotateArray(int[] array, int n) {
        for (int i = 0; i < n; i++) {
            int j, first;
            first = array[0];
            for (j = 0; j < array.length - 1; j++) {
                array[j] = array[j + 1];
            }
            array[j] = first;
        }
        return array;
    }
  

    /**
     * Adds AES round key
     *
     * @param state AES state
     * @param key   grouped key
     * @return state structure with rounded key
     */
    public List<List<Byte>> addRoundKey(List<List<Byte>> state, List<List<Byte>> key) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                List<Byte> stateNested = state.get(i);
                List<Byte> keyNested = key.get(i);

                Byte newValue = (byte) (stateNested.get(j) ^ keyNested.get(j));
                stateNested.set(j, newValue);
            }
        }
        return state;
    }

    /**
     * Subtraction of state structure bytes
     *
     * @param state AES state
     * @return subtracted state structure
     */
    public List<List<Byte>> subBytes(List<List<Byte>> state) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                List<Byte> stateNested = state.get(i);
                stateNested.set(j, (byte) this.sBox[stateNested.get(j) & 0xff]); //0xff for positive byte values
            }
        }
        return state;
    }

    /**
     * Shifts rows of state structure
     *
     * @param state AES state
     * @return shifted state structure
     */
    public List<List<Byte>> shiftRows(List<List<Byte>> state) {
        List<List<Byte>> newState = new ArrayList<>();
        newState.add(new ArrayList<>(4));
        newState.add(new ArrayList<>(4));
        newState.add(new ArrayList<>(4));
        newState.add(new ArrayList<>(4));

        int row = 0;
        int column = 0;
        for (int i = 0; i < 4; i++) {
            row = i;
            column = 0;
            for (int j = 0; j < 4; j++) {
                if (row == 4)
                    row = 0;
                newState.get(i).add(state.get(row).get(column));
                row++;
                column++;
            }
        }
        return newState;
    }

    /**
     * http://brandon.sternefamily.net/wp-content/uploads/2007/06/pyAES.txt
     * http://brandon.sternefamily.net/wp-content/uploads/2007/06/pyAES.txt
     *
     * @param state AES state
     * @return state structure with mixed columns
     */
    public List<List<Byte>> mixColumns(List<List<Byte>> state) {
        for (int i = 0; i < 4; i++) {
            List<Byte> column = new ArrayList<>();
            for (int j = 0; j < 4; j++) {
                byte i1 = state.get(i).get(j);
                column.add(i1);
            }

            column = mixColumn(column);

            for (int j = 0; j < 4; j++) {
                List<Byte> nestedState = state.get(i);
                nestedState.set(j, column.get(j));
            }
        }
        return state;
    }

    /**
     * Mixes column using Galois Field Multiplication
     *
     * @param column column
     * @return mixed column
     */
    public List<Byte> mixColumn(List<Byte> column) {
        List<Byte> columnCopy = new ArrayList<>(column);
        byte val0 = (byte) (galoisMul(columnCopy.get(0), (byte) 2) ^ galoisMul(columnCopy.get(3), (byte) 1) ^ galoisMul(columnCopy.get(2), (byte) 1) ^ galoisMul(columnCopy.get(1), (byte) 3));
        column.set(0, val0);
        byte val1 = (byte) (galoisMul(columnCopy.get(1), (byte) 2) ^ galoisMul(columnCopy.get(0), (byte) 1) ^ galoisMul(columnCopy.get(3), (byte) 1) ^ galoisMul(columnCopy.get(2), (byte) 3));
        column.set(1, val1);
        byte val2 = (byte) (galoisMul(columnCopy.get(2), (byte) 2) ^ galoisMul(columnCopy.get(1), (byte) 1) ^ galoisMul(columnCopy.get(0), (byte) 1) ^ galoisMul(columnCopy.get(3), (byte) 3));
        column.set(2, val2);
        byte val3 = (byte) (galoisMul(columnCopy.get(3), (byte) 2) ^ galoisMul(columnCopy.get(2), (byte) 1) ^ galoisMul(columnCopy.get(1), (byte) 1) ^ galoisMul(columnCopy.get(0), (byte) 3));
        column.set(3, val3);
        return column;
    }

    /**
     * Galois Field multiplication algorithm
     *
     * @param a value 1
     * @param b value 2
     * @return multiplication factor
     */
    public byte galoisMul(byte a, byte b) {
        int p = 0;
        for (int i = 0; i < 8; i++) {
            if ((b & 1) == 1) {
                p = p ^ a;
            }
            int hiBitSet = a & 0x80;
            a = (byte) ((a & 0xff) << 1); // ! a must be AND'ed with 0xff for positive value [fixed int length case]
            if (hiBitSet == 0x80) {
                a = (byte) (a ^ 0x1b);
            }
            b = (byte) (b >> 1);
        }
        return (byte) (p % 256);
    }


    /**
     * Creates new round key by choosing bits from expanded key
     *
     * @param expandedKey expanded key
     * @param round       round
     * @return new round key
     */
    public byte[] createRoundKey(byte[] expandedKey, int round) {
        List<Byte> newExpandedKey = new ArrayList<>();
        for (int i = (round * 16); i < (round * 16 + 16); i++) {
            newExpandedKey.add(expandedKey[i]);
        }
        return Bytes.toArray(newExpandedKey);
    }
}
