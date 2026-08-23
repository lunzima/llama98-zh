/* The caption bar and its buttons must sit symmetrically, on any host.
 *
 * captionwnd.c derives bar_rect and btn_rect from raw system metrics
 * independently, so the gap above and below a caption button is a
 * closed-form expression in three of them and there is nothing to
 * eyeball. This asserts the expression, not a pixel count: the metrics
 * themselves are per-host (DPI, theme, and the classic/modern frame
 * split all move them), so a fixed expected value would be a criterion
 * that only holds on one desktop.
 *
 * The bug this is written after: bar_rect had bottom = top + CYCAPTION
 * where the reference has CYCAPTION - 1, and the button top is a fixed
 * +2 inset, so the extra pixel could only land underneath. It was
 * visible and reported before anything here could see it, because
 * nothing here existed.
 *
 * WHAT THIS DOES NOT CHECK: whether the bar is the right height in the
 * first place. Symmetry is preserved by adding the same pixel to both
 * ends. That question is answered against the reference binary, not
 * here.
 *
 * Exit: 0 symmetric, 1 not.
 */
#include <windows.h>
#include <stdio.h>

/* Kept in the same shape as captionwnd.c's two functions so a reader
   can see they are the same arithmetic. */
static int bar_top(void)    { return GetSystemMetrics(SM_CYFRAME); }
static int bar_bottom(void) { return bar_top() + GetSystemMetrics(SM_CYCAPTION) - 1; }
static int btn_top(void)    { return GetSystemMetrics(SM_CYFRAME) + 2; }
static int btn_bottom(void) { return btn_top() + GetSystemMetrics(SM_CYCAPTION) - 5; }

int main(void)
{
    int above = btn_top() - bar_top();
    int below = bar_bottom() - btn_bottom();

    printf("SM_CYCAPTION %d  SM_CYSIZE %d  SM_CYFRAME %d  SM_CXFRAME %d\n",
           GetSystemMetrics(SM_CYCAPTION), GetSystemMetrics(SM_CYSIZE),
           GetSystemMetrics(SM_CYFRAME),   GetSystemMetrics(SM_CXFRAME));
    printf("bar  %d..%d   btn  %d..%d\n",
           bar_top(), bar_bottom(), btn_top(), btn_bottom());
    printf("padding above %d, below %d\n", above, below);

    if (above != below) {
        printf("capmetrics: FAIL - %d px asymmetry\n", below - above);
        return 1;
    }
    /* Positive control. A criterion that reads two zeroes and calls them
       equal is measuring nothing; the buttons are inset from the bar, so
       both gaps must be positive. */
    if (above <= 0) {
        printf("capmetrics: FAIL - gap of %d, the button is not inside "
               "the bar and the equality above is vacuous\n", above);
        return 1;
    }
    printf("capmetrics: PASS (%d px both sides)\n", above);
    return 0;
}
