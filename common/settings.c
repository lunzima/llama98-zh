/* Thinking mode and temperature. See settings.h for the four rules and
 * the per-parameter manual flags.
 */
#include "sampler.h"
#include "settings.h"
#include <stdio.h>      /* sprintf */

/* Format the "N tok · X tok/s" status cell for a turn that is (or just
 * was) generating.
 *
 * Pure arithmetic, so it can be driven with a
 * known token count and a known elapsed time and assert the formatted
 * rate - the same reason lz_common_ctx_clamp lives here rather than in
 * main.c. main.c feeds it the accumulated token count and a wall-clock
 * delta from lz_time_ms(), both of which change meaning across the
 * machines in the target family.
 *
 * `elapsed_ms` is interpreted the way cli_main.c reads the engine's own
 * timing: as a wall-clock span, seconds being the unit the number is
 * converted to. Anything under 1 ms counts as 0 ms, which keeps the
 * divisor away from 0 on a machine whose clock tick is coarse
 * (lz_time_ms is GetTickCount-based; see compat.h).
 *
 * The separator and unit are carried by the caller's FORMAT STRING, not
 * written here - gui/localized_strings.c owns the wording (the state
 * word is translated, the "tok"/"/s" unit deliberately is not; see
 * LZ_STR_STATE_TOKCELL). The format must contain "%d" and "%.1f" in
 * that order; a translation that dropped either is a bug.
 *
 * The format comes from the caller, not read here, because reading
 * lz_str_utf8 would drag gui/localized_strings.c into test_gui_settings
 * - a target that is deliberately free of Win32 and of the string table
 * that depends on it. The format is one constant in gui/main.c's call
 * site, next to the cell buffer. */
int lz_common_tokcell(char *out, int cap, int tokens, double elapsed_ms,
                   const char *fmt) {
    double secs;
    if (!out || cap <= 0) return 1;
    if (tokens < 0) tokens = 0;
    if (elapsed_ms < 1.0) elapsed_ms = 0.0;
    secs = elapsed_ms * 0.001;
    sprintf(out, fmt, tokens, secs > 0.0 ? tokens / secs : 0.0);
    return 0;
}

/* All three read the engine's own preset rather than a number written
   here - settings.h says why, and the three are separate functions
   rather than one returning a struct so that a caller asking for the
   default top_p cannot accidentally overwrite a temperature. */
float lz_common_settings_default_temp(int think) {
    LZSampleParams p;
    if (think) lz_sample_defaults_think(&p);
    else       lz_sample_defaults(&p);
    return p.temperature;
}

float lz_common_settings_default_topp(int think) {
    LZSampleParams p;
    if (think) lz_sample_defaults_think(&p);
    else       lz_sample_defaults(&p);
    return p.topp;
}

float lz_common_settings_default_rep(int think) {
    LZSampleParams p;
    /* Takes `think` even though both presets currently answer 1.1, so
       that the day they stop agreeing this function is already asking
       the right question and only settings.h's "no manual flag" note
       has to change. */
    if (think) lz_sample_defaults_think(&p);
    else       lz_sample_defaults(&p);
    return p.repetition_penalty;
}

float lz_common_settings_default_think_temp(int think) {
    LZSampleParams p;
    /* Both presets answer 0.3 today (sampler.c), which is exactly why
       the think toggle needs the manual flag: the value is invisible to
       the preset-following logic as long as the presets agree, and the
       day they stop agreeing a user who set one must not be silently
       overwritten. Read from the engine rather than hardcoded for the
       same reason the temperature's getter does. */
    if (think) lz_sample_defaults_think(&p);
    else       lz_sample_defaults(&p);
    return p.temp_think;
}

void lz_common_settings_init(LZGuiSettings *s) {
    if (!s) return;
    s->think = 1;                                 /* thinking starts on */
    /* GUI default is ON (flag clear, value the preset's); the CLI is
       the opt-in side. */
    s->think_temp = lz_common_settings_default_think_temp(1);
    s->manual_think_temp = 0;
    s->temp = lz_common_settings_default_temp(1);
    s->manual_temp = 0;
    s->topp = lz_common_settings_default_topp(1);
    s->manual_topp = 0;
    s->rep = lz_common_settings_default_rep(1);
    /* Unlimited, spelled -1 rather than 0: 0 would read as "zero
       tokens" in the ini and the dialog (see settings.h). */
    s->max_new = LZ_COMMON_MAXNEW_UNLIMITED;
    /* Empty: the built-in identity prompt stays in force, and its TEXT
       is never copied in here (settings.h - the GUI reads it live from
       lz_chat_default_system() instead). */
    s->system[0] = '\0';
    s->seed_mode = LZ_COMMON_SEED_RANDOM;             /* see settings.h */
    s->seed = 1;
    s->ctx = LZ_COMMON_CTX_DEFAULT;
}

int lz_common_ctx_clamp(int want, int model_cap) {
    if (want < LZ_COMMON_CTX_MIN) want = LZ_COMMON_CTX_MIN;
    if (want > LZ_COMMON_CTX_MAX) want = LZ_COMMON_CTX_MAX;
    /* Second, and only downward: a model that advertises MORE than
       LZ_COMMON_CTX_MAX does not raise the ceiling, because the ceiling is
       about this machine's memory and not about the model. */
    if (model_cap > 0 && want > model_cap) want = model_cap;
    return want;
}

int lz_common_topp_to_scroll(float t) {
    int pos;
    if (t < LZ_COMMON_TOPP_MIN) t = LZ_COMMON_TOPP_MIN;
    if (t > LZ_COMMON_TOPP_MAX) t = LZ_COMMON_TOPP_MAX;
    pos = (int)(t * 100.0f + 0.5f);
    /* Not redundant after the clamps above, for the same reason the
       temperature's pair says: a NaN fails BOTH comparisons (every
       comparison with NaN is false) and reaches the cast, where the
       conversion is undefined. */
    if (pos < LZ_COMMON_TOPP_SCROLL_MIN) pos = LZ_COMMON_TOPP_SCROLL_MIN;
    if (pos > LZ_COMMON_TOPP_SCROLL_MAX) pos = LZ_COMMON_TOPP_SCROLL_MAX;
    return pos;
}

float lz_common_scroll_to_topp(int pos) {
    if (pos < LZ_COMMON_TOPP_SCROLL_MIN) pos = LZ_COMMON_TOPP_SCROLL_MIN;
    if (pos > LZ_COMMON_TOPP_SCROLL_MAX) pos = LZ_COMMON_TOPP_SCROLL_MAX;
    return (float)pos * 0.01f;
}

int lz_common_rep_to_scroll(float r) {
    int pos;
    if (r < LZ_COMMON_REP_MIN) r = LZ_COMMON_REP_MIN;
    if (r > LZ_COMMON_REP_MAX) r = LZ_COMMON_REP_MAX;
    pos = (int)((r - LZ_COMMON_REP_MIN) * (1.0f / LZ_COMMON_REP_STEP) + 0.5f);
    if (pos < 0) pos = 0;
    if (pos > LZ_COMMON_REP_SCROLL_MAX) pos = LZ_COMMON_REP_SCROLL_MAX;
    return pos;
}

float lz_common_scroll_to_rep(int pos) {
    if (pos < 0) pos = 0;
    if (pos > LZ_COMMON_REP_SCROLL_MAX) pos = LZ_COMMON_REP_SCROLL_MAX;
    return LZ_COMMON_REP_MIN + (float)pos * LZ_COMMON_REP_STEP;
}

void lz_common_settings_set_think(LZGuiSettings *s, int think) {
    if (!s) return;
    s->think = think ? 1 : 0;
    /* Two flags, two independent decisions. One shared flag here would
       mean a user who set a temperature also froze their top_p at the
       old mode's default - see LZGuiSettings' own comment. */
    if (!s->manual_temp) s->temp = lz_common_settings_default_temp(s->think);
    if (!s->manual_topp) s->topp = lz_common_settings_default_topp(s->think);
    /* Same rule, third flag. Both presets answer 0.3 today so this never
       visibly moves, but keeping it here is what makes the next preset
       change not a silent overwrite of a value the user set. */
    if (!s->manual_think_temp)
        s->think_temp = lz_common_settings_default_think_temp(s->think);
}

int lz_common_settings_set_temp(LZGuiSettings *s, float t) {
    if (!s) return 1;
    /* The `!(t >= 0)` spelling rather than `t < 0` is deliberate: it also
       rejects a NaN, which a text field can produce and which would
       otherwise sail through both comparisons and poison the sampler. */
    if (!(t >= 0.0f) || t > LZ_COMMON_TEMP_MAX) return 1;
    s->temp = t;
    s->manual_temp = 1;
    return 0;
}

int lz_common_settings_set_think_temp(LZGuiSettings *s, float t) {
    if (!s) return 1;
    /* Same guard as set_temp: the `!(t >= 0)` spelling rejects a NaN,
       and a NaN reaching the sampler would poison every think block. */
    if (!(t >= 0.0f) || t > LZ_COMMON_TEMP_MAX) return 1;
    s->think_temp = t;
    s->manual_think_temp = 1;
    return 0;
}

int lz_common_settings_set_topp(LZGuiSettings *s, float t) {
    if (!s) return 1;
    if (!(t >= LZ_COMMON_TOPP_MIN) || t > LZ_COMMON_TOPP_MAX) return 1;
    s->topp = t;
    s->manual_topp = 1;
    return 0;
}

int lz_common_settings_set_rep(LZGuiSettings *s, float r) {
    if (!s) return 1;
    if (!(r >= LZ_COMMON_REP_MIN) || r > LZ_COMMON_REP_MAX) return 1;
    s->rep = r;
    return 0;
}

int lz_common_settings_set_max_new(LZGuiSettings *s, int n, int ctx_cap) {
    if (!s) return 1;
    /* Any negative is unlimited, not a refusal - see settings.h. The
       normalisation happens HERE and only here, so nothing downstream
       ever has to decide what -7 means. */
    if (n < 0) { s->max_new = LZ_COMMON_MAXNEW_UNLIMITED; return 0; }
    if (n < LZ_COMMON_MAXNEW_MIN) return 1;
    if (ctx_cap > 0 && n > ctx_cap) return 1;
    s->max_new = n;
    return 0;
}

void lz_common_settings_restore(LZGuiSettings *s) {
    if (!s) return;
    s->think_temp = lz_common_settings_default_think_temp(s->think);
    s->manual_think_temp = 0;
    s->temp = lz_common_settings_default_temp(s->think);
    s->manual_temp = 0;
    s->topp = lz_common_settings_default_topp(s->think);
    s->manual_topp = 0;
    s->rep = lz_common_settings_default_rep(s->think);
    s->max_new = LZ_COMMON_MAXNEW_UNLIMITED;
    s->ctx = LZ_COMMON_CTX_DEFAULT;
}
