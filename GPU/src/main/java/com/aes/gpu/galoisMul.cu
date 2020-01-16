extern "C"
__global__ void add(int n, unsigned char *a, unsigned char *b, unsigned char *res)
{
     int p = 0;
        for (int i = 0; i < 8; i++) {
            if ((*b & 1) == 1) {
                p = p ^ *a;
            }
            int hiBitSet = *a & 0x80;
            *a = (unsigned char)((*a & 0xff) << 1); // ! a must be AND'ed with 0xff for positive value [fixed int length case]
            if (hiBitSet == 0x80) {
                *a = (unsigned char)(*a ^ 0x1b);
            }
            *b = (unsigned char)(*b >> 1);

        }
        *res = (unsigned char)(p % 256);
}