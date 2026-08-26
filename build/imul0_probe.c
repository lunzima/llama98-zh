/* Two loop shapes, same result: tk*nb strength reduction vs a hand-written
   pointer increment. If the pointer-increment form drops the imul-by-zero,
   that is the rewrite for the engine's row-kernel setups. */
typedef int int32_t;

int32_t *g_a;

void loop_tk_mul(int32_t *acc, int nt, int nb) {
    int tk;
    for (tk = 0; tk < nt; tk++) {
        g_a = acc + (unsigned int)tk * nb;
    }
}

void loop_ptr_inc(int32_t *acc, int nt, int nb) {
    int tk;
    int32_t *a = acc;
    for (tk = 0; tk < nt; tk++, a += nb) {
        g_a = a;
    }
}
