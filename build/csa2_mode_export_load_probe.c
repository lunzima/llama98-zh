#include <stdio.h>
#include "model.h"

int main(int argc, char **argv) {
    LZModel m;
    char errbuf[512];
    if (argc < 2) { printf("usage: probe <dir>\n"); return 2; }
    if (lz_open(&m, argv[1], errbuf, sizeof errbuf) != 0) {
        printf("FAIL: %s\n", errbuf);
        return 1;
    }
    printf("PASS: loaded OK\n");
    return 0;
}
