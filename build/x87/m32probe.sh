#!/bin/sh
# Which compiler on this box can LINK -m32, not merely compile it.
#
# Ubuntu ships multilib per compiler version, so `gcc` being gcc-13
# while gcc-14-multilib is what is installed makes a compile-only probe
# say yes and the link say "cannot find -lgcc". Probing with a link is
# the whole point; build/ci.sh stage 9 does the same and passes the
# winner through as CC.
for c in gcc gcc-14 gcc-13 gcc-12; do
    command -v "$c" >/dev/null 2>&1 || continue
    printf '%-8s ' "$c"
    echo 'int main(void){return 0;}' > /tmp/m32probe.c
    if "$c" -m32 /tmp/m32probe.c -o /tmp/m32probe 2>/dev/null; then
        echo links
    else
        echo no
    fi
done
rm -f /tmp/m32probe.c /tmp/m32probe
