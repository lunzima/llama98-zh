#ifndef LZ_GUI_MIDIPLAY_H
#define LZ_GUI_MIDIPLAY_H

#include <windows.h>

/* Playing one MIDI file, at a tempo the caller moves while it plays.
 *
 * Three layers: common/smf.c turns bytes into events (no Win32),
 * gui/compat40.c makes the winmm calls (the only file that may name a
 * 4.0-era type or resolve an entry point at run time), and this one
 * decides which piece, whether to loop, and what tempo.
 *
 * ONE piece at a time: the target has one MIDI device, so the chime
 * replaces the music rather than mixing with it.
 */

/* 1 when a stream can be opened. Opens the device and keeps it open -
 * the target takes a noticeable moment over it. */
int lz_midi_system_ok(HWND notify);

/* 0 until the bank's compressed audio format is finished. Answering
 * honestly is what lets the sound window mark the row. */
int lz_midi_wave_ok(void);

/* Load and start `path`. `loop` replays until stopped; the chime
 * passes 0. `ratio_milli` 1000 is the written tempo.
 *
 * Every failure - no device, no file, not MIDI, too big - answers 0
 * with nothing playing. None of them is worth a message box. */
int lz_midi_start(const char *path, int loop, int instrument,
                  int volume, int ratio_milli);

/* The finish chime, from the bytes built into common/chime.c. No path
 * and no file: an installation where music/ was never copied still
 * says the reply has landed. */
int lz_midi_start_chime(int instrument, int volume);

/* Stop, and silence anything still sounding. */
void lz_midi_halt(void);

/* Record a new speed ratio. Nothing is sent from here: the ratio is
 * carried as a MEVT_TEMPO at the head of the next chunk, which is the
 * only place the stream can be re-clocked, so the move costs what a
 * submission costs and this call is a store - cheap enough for every
 * status-bar tick. */
void lz_midi_ratio(int ratio_milli);

/* A position reading with NO submission between the two samples: the
 * control for what a `tempo` line's bracket shows. Called from the
 * same place lz_midi_ratio is, on the ticks where the ratio did not
 * move, so the arms are sampled at one cadence. It writes nothing
 * and plays nothing either way - without a trace it returns on its
 * first test. */
void lz_midi_probe(void);

/* The sound settings changed while a piece was playing: STOP.
 *
 * Nothing can be changed under a piece that is already queued - the
 * buffer carries the volume and the patch at tick 0, so a controller
 * sent alongside the stream would hold only until the loop put the old
 * values back. Restarting the piece was the other candidate and it
 * worked, but a restart mid-phrase is heard as the music jumping
 * backwards. Stopping says one thing plainly, and the next turn starts
 * with the new settings. */
void lz_midi_settings_changed(void);

int  lz_midi_busy(void);

/* From the window procedure, on LZ_UI_MM_MOM_DONE. */
void lz_midi_buffer_done(void);

void lz_midi_shutdown(void);

/* ---- a trace of the stream, off unless a file is named for it ----
 *
 * One line per stream event - a start, a chunk handed to the device (a
 * `tempo` line when it carried a tempo event and a `chunk` line when it
 * did not), a buffer completion, a loop, a halt - written to `path`,
 * which gui/main.c takes from the ini key `music_trace`. An empty path,
 * or one that cannot be opened, leaves the trace OFF; a key naming a
 * file the program cannot write is the one failure nothing reports, so
 * it looks exactly like a trace that had nothing to say.
 *
 * OFF costs one test per event: no file, no formatting, and no call to
 * the device or to the clock either. ON changes nothing about playback.
 * What it adds is bounded and named: two position queries and two
 * counter reads around each submission - the bracket `pos0`/`pos1`/
 * `dt_us` every chunk line carries - and the same pair for a `probe`
 * line, the control reading taken where the ratio did not move, at most
 * one every LZ_MIDI_PROBE_MIN_US (midiplay.c holds that bound). The
 * file's own header
 * says what a tick is, why a chunk's tick range is comparable with
 * `pos` only inside one playback, and why the `t` column cannot
 * separate two events 100 ms apart on a machine that has been up for
 * days. */
void lz_midi_trace_to(const char *path);

#endif /* LZ_GUI_MIDIPLAY_H */
