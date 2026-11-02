/* Force-included into the engine TUs that carry LZ_TAP call sites, so a
 * -D on the command line can point those sites at the probe's dump
 * function. Declares only; build/kunmoe_hidden_probe.c defines it.
 *
 * ENGINE-SIDE INSTRUMENT, next to the probe it belongs to - not a
 * training tool, so it lives in this tree's build/ directory. BUILD/ IS
 * GITIGNORED: commit it with `git add -f`, as the other gates here are.
 */
#ifndef LZ_KUNMOE_HIDDEN_TAP_H
#define LZ_KUNMOE_HIDDEN_TAP_H
void lz_dump_tap(const char *tag, int li, const float *v, int n);
#endif
