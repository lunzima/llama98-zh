#ifndef LZ_GUI_SETTINGS_H
#define LZ_GUI_SETTINGS_H

/* Thinking mode and temperature, and the coupling between them.
 *
 * Four rules govern the coupling, and they are the kind of small that
 * goes wrong quietly - a user who set a temperature and then toggled
 * think does not expect their number to be replaced, and nothing about
 * the window would tell them it had been.
 *
 * The numbers themselves are NOT here. They come from the engine
 * (lz_sample_defaults / lz_sample_defaults_think) because a front end
 * with its own copy gives different output from the CLI on the same
 * model, and the two drift apart the first time either is tuned.
 *
 * A FLAG, not "current value != the old default", decides whether the
 * user has set a temperature. The value-comparison test is wrong in
 * exactly one case: a user who deliberately types the number that
 * happens to BE the default is then treated as never having touched it,
 * and their value gets replaced on the next think toggle. A flag says
 * what the rule actually means - "I set this, it stays" - and cannot be
 * fooled that way.
 */

/* Seed policy. RANDOM is the default. A caller that never assigns
 * rng_seed would silently run every session on seed 1 (the sampler
 * substitutes 1 for a seed of 0) - two such launches with the same
 * prompt would produce byte-identical replies. cli_main.c randomises
 * when the seed is 0 and the HTTP endpoint uses a fixed 20260805; the
 * GUI defaults to LZ_COMMON_SEED_RANDOM. */
#define LZ_COMMON_SEED_RANDOM 0
#define LZ_COMMON_SEED_FIXED  1

/* Custom system prompt cap. Generous for a boxful of context; the
 * engine's prompt is built at runtime so the cap is about the dialog
 * and the ini, not about memory. Declared before the struct, which
 * embeds a char[LZ_COMMON_SYSTEM_MAX+1]. */
#define LZ_COMMON_SYSTEM_MAX 4096

typedef struct {
    int   think;        /* 1 = thinking mode on */
    /* think-block dynamic temperature. Follows the thinking preset like
       temp/topp do, so it needs its own manual flag for the
       same reason: a user who sets it must not have it overwritten by a
       think toggle. GUI defaults it ON (manual flag 0 but enabled),
       unlike the CLI where it is opt-in. */
    float think_temp;
    int   manual_think_temp;
    float temp;         /* current temperature, always within [0, cap] */
    /* ONE FLAG PER PRESET-FOLLOWING PARAMETER, never a shared one.
       Both temperature and top_p differ between the two presets
       (0.6/0.8 and 0.8/0.95), and both must keep a value the user typed
       across a think toggle. With a single `manual` covering both, a
       user who touches the temperature also freezes their top_p at
       whatever the old mode's default was - one question ("has the user
       set this?") with two right answers and one place to store them. */
    int   manual_temp;
    float topp;         /* nucleus sampling, within [MIN, MAX] */
    int   manual_topp;
    /* Repetition penalty. NO manual flag, and that is not an omission:
       both engine presets set 1.1, so there is no preset for it to
       follow and nothing a think toggle could overwrite. If the two
       presets ever disagree, this needs a flag the same day. */
    float rep;
    /* Generation cap. -1 is "unlimited"; see LZ_COMMON_MAXNEW_UNLIMITED. */
    int   max_new;
    /* Custom system prompt.
     *
     * UTF-8, and it is NEVER the engine's built-in default. Semantics:
     *   empty  -> no system message is pushed, and the engine's
     *             Item 9b injects the trained constant -> byte-identical
     *             to today's behaviour.
     *   non-empty -> rendered at the TOP of every conversation, in place
     *             of the built-in (REPLACE). Emptying the box restores
     *             the built-in.
     *
     * The built-in's TEXT is never copied into this field - the GUI
     * reads it live from lz_chat_default_system() when it needs to show
     * it, so there is exactly one copy of the constant (src/chat.c).
     * "Restore defaults" clears this field rather than re-typing the
     * constant, which is the whole point of the getter. */
    char  system[LZ_COMMON_SYSTEM_MAX + 1];
    int   seed_mode;    /* LZ_COMMON_SEED_* */
    /* Only read when seed_mode is FIXED. 0 is accepted and means the
       same thing it means to the engine (the sampler substitutes 1),
       so a user who types 0 gets a working fixed seed rather than a
       silently random one. */
    unsigned long long seed;
    /* Requested context window, in tokens, always a multiple of
       LZ_COMMON_CTX_STEP within [MIN, MAX]. This is what the USER asked
       for; what a loaded model's run state was actually allocated with
       is LZGuiModel.seq_len, and the two differ whenever the model's
       own cfg->seq_len is the smaller number. Read the second one when
       the question is "how much context is there", the first only when
       the question is "what did the user pick". */
    int   ctx;
} LZGuiSettings;

/* Context window range. The range exists because a fixed size is the
 * one thing a 128 MB machine and a 1 GB machine cannot agree on;
 * LZ_COMMON_CTX_DEFAULT is the default.
 *
 * The ceiling is 32768. The real ceiling is the model's own
 * max_position_embeddings - 32768 on KunMoe - and lz_common_ctx_clamp
 * applies that cap per-model, so the hard GUI ceiling only stops a value
 * the model could never use either; the per-model cap is still what
 * actually bounds the allocation. */
#define LZ_COMMON_CTX_MIN     512
#define LZ_COMMON_CTX_MAX     32768
#define LZ_COMMON_CTX_STEP    512
#define LZ_COMMON_CTX_DEFAULT 2048

/* Clamp a requested context to what is actually allocatable: into
 * [MIN, MAX] first, then down to the model's own cfg->seq_len.
 *
 * `model_cap <= 0` means "no model, so no cap" rather than "cap of
 * zero" - the settings dialog is reachable before anything is loaded,
 * and a zero there must not collapse every request to the floor.
 *
 * Without the model cap a user who picks 32768 on a model trained to
 * 4096 gets an allocation that succeeds, a KV cache twice the size it
 * can ever use, and RoPE positions the model never saw.
 *
 * Pure: no Win32, no engine, so every interesting pair can be tabled.
 * The result is NOT re-rounded to any step -
 * a model cap of 3000 gives 3000, because the point is what the
 * allocator will be handed, not what a slider can display. */
int lz_common_ctx_clamp(int want, int model_cap);

/* A hard ceiling, not a suggestion. Small models above it do not get
 * more creative, they stop answering the question. */
#define LZ_COMMON_TEMP_MAX 1.0f

/* top_p. The floor is 0.05 rather than 0: nucleus sampling at 0 keeps
 * nothing, and the engine's own behaviour there is not something a
 * settings dialog should be the first to explore. */
#define LZ_COMMON_TOPP_MIN 0.05f
#define LZ_COMMON_TOPP_MAX 1.0f

/* Repetition penalty. 1.0 is the identity and is a legal choice, which
 * is why the floor is 1.0 and not something above it; the ceiling is
 * where a 57.6M model starts refusing to reuse words it needs. */
#define LZ_COMMON_REP_MIN  1.0f
#define LZ_COMMON_REP_MAX  1.5f
#define LZ_COMMON_REP_STEP 0.05f

/* Maximum new tokens.
 *
 * -1 AND 0 ARE THE SAME THING TO THE ENGINE - llama_zh.h's field
 * comment is "<=0 means use the model's seq_len". Choosing -1 is a
 * SPELLING decision, not a behaviour one: it makes "unlimited" look
 * like unlimited in the ini and in the dialog instead of looking like
 * "zero tokens". Nobody should later "normalise" it to 0, and nobody
 * should read 0 as "generate nothing".
 *
 * "Unlimited" does not mean "will not stop": EOS and the stop strings
 * still apply (gui/session.c sets a set), so the real meaning is "until
 * EOS, a stop string, the context filling, or the user pressing Stop".
 *
 * The floor for a FINITE value is 16 - below that the cap fires inside
 * the model's own preamble and the reply is a fragment, which reads as
 * a broken model rather than as a setting. */
#define LZ_COMMON_MAXNEW_UNLIMITED (-1)
#define LZ_COMMON_MAXNEW_MIN       16

/* Thinking starts ON; temp and topp are that mode's engine defaults and
 * neither manual flag is set. */
void lz_common_settings_init(LZGuiSettings *s);

/* The engine's defaults for a think state. */
float lz_common_settings_default_temp(int think);
float lz_common_settings_default_topp(int think);
float lz_common_settings_default_rep(int think);
/* Think-block dynamic temperature - same contract as the other three:
   read from the engine's preset, not a number copied here. */
float lz_common_settings_default_think_temp(int think);

/* Toggle thinking. Temperature and top_p each follow the new mode's
 * default ONLY while the user has not set that one - separately, which
 * is what the two flags are for. */
void lz_common_settings_set_think(LZGuiSettings *s, int think);

/* Set the temperature. Returns 0 on success; non-zero when the value is
 * out of range, in which case NOTHING changes - a refusal, not a clamp.
 * A clamp turns "1.5" into a silently
 * different request; a refusal tells the user the ceiling exists. */
int lz_common_settings_set_temp(LZGuiSettings *s, float t);

/* Same contract as set_temp - refuse, do not clamp, change nothing on
   refusal - for the think-block temperature. Shares the temperature's
   own [0, LZ_COMMON_TEMP_MAX] ceiling; marks manual_think_temp on
   success. */
int lz_common_settings_set_think_temp(LZGuiSettings *s, float t);

/* Same contract as set_temp - refuse, do not clamp, change nothing on
 * refusal. set_topp additionally marks manual_topp, set_rep does not
 * mark anything because there is nothing for it to resist. */
int lz_common_settings_set_topp(LZGuiSettings *s, float t);
int lz_common_settings_set_rep(LZGuiSettings *s, float r);
/* Accepts LZ_COMMON_MAXNEW_UNLIMITED, or [LZ_COMMON_MAXNEW_MIN, ctx_cap].
 * `ctx_cap` is the context window currently in force; pass 0 for "no
 * cap known yet", which accepts anything at or above the floor. ANY
 * negative value is stored as -1 rather than refused - the dialog and
 * the ini both offer "unlimited" and there is no second meaning a
 * negative number could carry. */
int lz_common_settings_set_max_new(LZGuiSettings *s, int n, int ctx_cap);

/* top_p and the repetition penalty <-> horizontal scrollbar position.
 * Same pairing as the temperature's, and the same division of labour:
 * the slider is the coarse picker, the value box takes any number in
 * range. The scroll RANGE differs per control, which is why each has
 * its own pair rather than everything sharing 0..100 - a repetition
 * penalty has eleven useful positions, not a hundred. */
#define LZ_COMMON_TOPP_SCROLL_MIN 5      /* 0.05 * 100 */
#define LZ_COMMON_TOPP_SCROLL_MAX 100
int   lz_common_topp_to_scroll(float t);
float lz_common_scroll_to_topp(int pos);

#define LZ_COMMON_REP_SCROLL_MAX 10      /* (1.5 - 1.0) / 0.05 */
int   lz_common_rep_to_scroll(float r);
float lz_common_scroll_to_rep(int pos);

/* Back to the current think mode's defaults, and forget that the user
 * ever set any of them. EVERY setting this struct carries, not only the
 * temperature - "restore defaults" that leaves one of the defaults
 * alone is the kind of half-answer nobody notices until they are
 * hunting for why the program still behaves oddly. */
void lz_common_settings_restore(LZGuiSettings *s);

/* Format the "N tok · X tok/s" status cell. Pure arithmetic:
 * no Win32, so known inputs can be tabled and
 * the formatted rate asserted. `elapsed_ms` is a wall-clock span, in the
 * same seconds-as-the-unit convention cli_main.c uses; anything under
 * 1 ms is treated as 0 ms (see common/settings.c). `fmt` is the
 * caller-supplied localised format string ("%d" then "%.1f") - it is a
 * parameter rather than read in here so this file stays free of the
 * Win32-dependent string table. Returns 0 on success. */
int lz_common_tokcell(char *out, int cap, int tokens, double elapsed_ms,
                   const char *fmt);

#endif
