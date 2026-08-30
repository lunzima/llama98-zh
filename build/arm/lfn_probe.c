#include <stdio.h>
#include <string.h>
#include "lfn.h"
int main(int argc, char **argv){
    const char *d = argc>1 ? argv[1] : ".";
    int e = lz_lfn_exists(d, "model.bin");
    printf("exists model.bin in %s: %d\n", d, e);
    return 0;
}
