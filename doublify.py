import random
import sys
import os
import logging

log = logging.getLogger(__name__)

def parse_sm(sm):
    pairs = []
    i = 0
    while True:
        hash_idx = sm.find(b'#', i)
        if hash_idx == -1:
            break

        colon_idx = sm.find(b':', hash_idx)

        if colon_idx == -1:
            break

        semicolon_idx = sm.find(b';', colon_idx)

        if semicolon_idx == -1:
            break

        if hash_idx > i:
            pairs.append((None, sm[i:hash_idx]))

        key = sm[hash_idx+1:colon_idx]

        value = sm[colon_idx+1:semicolon_idx]
        
        pairs.append((key, value))
        
        i = semicolon_idx+1

    # Append remainder.
    if i < len(sm):
        pairs.append((None, sm[i:]))

    return pairs


def encode_sm(pairs):
    chunks = []
    for key, value in pairs:
        if key is None:
            chunks.append(value)
        else:
            chunks.append(b'#' + key + b':' + value + b';')

    return b''.join(chunks)


allowed_lr_pairs = set([
    (0,1), (0,2), (0,3),
    (1,2), (1,3), (1,4),
    (2,1), (2,3), (2,4),
    (3,4), (3,5), (3,6),
    (4,5), (4,6), (4,7),
    (5,6), (5,7),
    (6,5), (6,7),
])

allowed_foot_movements = set([
    (0,0), (0,1), (0,2),
    (1,0), (1,1), (1,2), (1,3),
    (2,0), (2,1), (2,2), (2,3),
    (3,1), (3,2), (3,3), (3,4),
    (4,3), (4,4), (4,5), (4,6),
    (5,4), (5,5), (5,6), (5,7),
    (6,4), (6,5), (6,6), (6,7),
    (7,5), (7,6), (7,7),
])

left_foot_distances = [
    [0, 0, 0, 1, 2, 3, 3, 4],
    [1, 0, 0, 0, 1, 2, 2, 3],
    [2, 1, 1, 0, 0, 1, 1, 2],
    [3, 2, 2, 1, 0, 0, 0, 1],
]

right_foot_distances = [
    [1, 0, 0, 0, 1, 2, 2, 3],
    [2, 1, 1, 0, 0, 1, 1, 2],
    [3, 2, 2, 1, 0, 0, 0, 1],
    [4, 3, 3, 2, 1, 0, 0, 0],
]
positions = [0,1,2,3,2,1]

jumps_for_position = [
    [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)],
    [(1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
    [(3, 4), (3, 5), (3, 6), (4, 5), (4, 6)],
    [(4, 5), (4, 6), (4, 7), (5, 7), (6, 7)],
]

NEVER = float('-inf')
STATE_STEP_COUNT = 7

FORWARDNESS = [1, 0, 2, 1, 1, 0, 2, 1]

def rate_step(notes, is_left_foot, position_index, is_single_step, og_note):
    # Check position
    position = positions[position_index]
    if is_left_foot:
        dist = left_foot_distances[position][notes[-1]]
    else:
        dist = right_foot_distances[position][notes[-1]]
    if dist != 0:
        return NEVER

    # Start in middle
    if len(notes) == 1:
        if is_left_foot:
            if notes[0] != 3: return NEVER
        else:
            if notes[0] != 4: return NEVER

    if len(notes) == 2:
        if notes != (3,4) and notes != (4,3):
            return NEVER

    # Check stretch / crossover
    if len(notes) >= 2:
        a, b = notes[-2:]
        if is_left_foot:
            a, b = b, a
        if not (a, b) in allowed_lr_pairs:
            return NEVER

    # Check shmoove
    if len(notes) >= 3:
        a = notes[-1]
        b = notes[-3]
        if not (a, b) in allowed_foot_movements:
            return NEVER

    # Check fast transition
    if len(notes) >= 5:
        a = notes[-1]
        b = notes[-5]
        if not (a, b) in allowed_foot_movements:
            return NEVER

    score = 0

    # Avoid medium-fast transition
    if len(notes) >= 7:
        a = notes[-1]
        b = notes[-7]
        if not (a, b) in allowed_foot_movements:
            score -= 0.15

    if len(notes) >= 3:
        is_step = notes[-1] == notes[-3]
        was_step = len(notes) >= 5 and notes[-3] == notes[-5]
        was_was_step = len(notes) >= 7 and notes[-5] == notes[-7]
        step_matches = is_step == is_single_step

        # Prefer matching is_single_step.
        if step_matches:
            score += 1

        # Slightly prefer mixing steps & non-steps.
        if is_step == was_step:
            score -= 0.1
            if was_step == was_was_step:
                score -= 0.1

    # Avoid crossing the middle a lot
    if len(notes) >= 7:
        a = notes[-1]
        b = notes[-3]
        c = notes[-5]
        d = notes[-7]

        if all(n in (3,4) for n in (a,b,c,d)):
            if a != b and b != c and c != d:
                score -= 0.8

    # Try to match up/down
    if len(notes) >= 1:
        if FORWARDNESS[og_note] == FORWARDNESS[notes[-1]]:
            score += 0.1

    # Slightly penalize having a foot avoid the middle when in middle positions.
    if len(notes) >= 7 and position in [1,2]:
        a = notes[-1]
        b = notes[-3]
        c = notes[-5]
        d = notes[-7]

        if all(FORWARDNESS[n] in (0,2) for n in (a,b,c,d)):
            score -= 0.05

    return score


JACK = 1
STEP = 2
SLIDE = 3
JUMP = 4
JUMP_JACK = 5

def get_step_pattern(measures):
    step_pattern = []

    prev_notes = [None, None]

    for measure_index, measure in enumerate(measures):
        for notes_index, notes in enumerate(measure):
            note_count = sum(n in b'124' for n in notes)
            step_kind = None
            og_note = None
            if note_count >= 2:
                if notes == prev_notes[-1]:
                    step_kind = JUMP_JACK
                else:
                    step_kind = JUMP
            elif note_count == 1:
                og_note = max(notes.find(n) for n in b'124')
                if notes == prev_notes[-1]:
                    step_kind = JACK
                elif notes == prev_notes[-2]:
                    step_kind = STEP
                else:
                    step_kind = SLIDE

            if step_kind is not None:
                note_beat = 4*measure_index + 4*notes_index/len(measure)
                step_pattern.append((step_kind, og_note, note_beat))
                prev_notes = [prev_notes[-1], notes]

    return step_pattern

def beats_to_times(bpm_changes, beats):
    bpm_change_index = 0
    prev_beat = 0
    time = 0
    times = []
    for beat in beats:
        # Process bpm changes between prev_beat and beat.
        while True:
            if bpm_change_index + 1 == len(bpm_changes):
                break
            if bpm_changes[bpm_change_index + 1][0] > beat:
                break
            change_beat = bpm_changes[bpm_change_index + 1][0]
            time += (change_beat - prev_beat) * 60 / bpm_changes[bpm_change_index][1]
            prev_beat = change_beat
            bpm_change_index += 1
        # Advance time to beat
        time += (beat - prev_beat) * 60 / bpm_changes[bpm_change_index][1]
        times.append(time)
        prev_beat = beat

    return times

def transition_step_bounds(bpm):
    # Don't worry.
    lo_bpm = 100
    mid_bpm = 190
    hi_bpm = 240

    lo_min = 8
    lo_max = 12

    mid_min = 16
    mid_max = 32

    hi_min = 24
    hi_max = 32

    if bpm <= lo_bpm:
        return (lo_min, lo_max)
    if bpm <= mid_bpm:
        t = (bpm - lo_bpm) / (mid_bpm - lo_bpm)
        return (lo_min + (mid_min-lo_min)*t, lo_max + (mid_max-lo_max)*t)
    if bpm <= hi_bpm:
        t = (bpm - mid_bpm) / (hi_bpm - mid_bpm)
        return (mid_min + (hi_min-mid_min)*t, mid_max + (hi_max-mid_max)*t)

    return (hi_min, hi_max)

BEAM_SIZE = 50
def doublify_measures(measures, bpm_changes):
    double_stream = []

    step_pattern = get_step_pattern(measures)

    beats = [p[2] for p in step_pattern]
    times = beats_to_times(bpm_changes, beats)

    times.append(times[-1] + 1)

    durations = [times[i+1] - times[i] for i in range(len(times)-1)]
    bpms = [60/4/d for d in durations]

    # We want to find the chart that maximizes total score according to rate_step.
    # But there's exponentially many possible charts to score.
    # So I'm using beam search to trim the possibilities (https://en.wikipedia.org/wiki/Beam_search)

    # States:
    # notes, is_left_foot -> score, full steps
    # full steps is a linked list (step, tail) of steps in reverse order.
    # This makes it possible to discard equivalent states with lower score.
    states = {
        ((), bool(random.randint(0, 1))): (0, None),
    }

    position_rate = 0 # 0 - 1, how fast the position changes.
    position_progress = 0 # 0 - 1, progress until next transition.
    # Always start in one of the middle positions, moving towards the same side of that position.
    position_index = random.choice([2,5])
    was_jump = False

    for note_index in range(len(step_pattern)):
        note_kind, og_note, _ = step_pattern[note_index]
        bpm = bpms[note_index]
        if note_kind in [STEP, SLIDE] and not was_jump:
            if position_progress >= 1:
                position_index = (position_index + 1) % len(positions)
                position_progress = 0
                position_rate = random.random()

        pos_steps_lo, pos_steps_hi = transition_step_bounds(bpm)
        pos_steps = pos_steps_lo + position_rate*(pos_steps_hi - pos_steps_lo)
        position_progress += 1 / pos_steps

        new_state_pairs = []
        for old_state_key, old_state_value in states.items():
            old_steps, old_left_foot = old_state_key
            old_score, old_full_steps = old_state_value

            if note_kind == JACK:
                new_state_pairs.append((
                    (old_steps, old_left_foot),
                    (old_score, (old_full_steps[0], old_full_steps)),
                ))
            elif note_kind == JUMP and was_jump:
                old1 = old_full_steps[0]
                old0 = old_full_steps[1][0]

                candidates = [
                    j for j in jumps_for_position[positions[position_index]]
                    if min(*j) != min(old0, old1) or max(*j) != max(old0, old1)
                ]
                note0, note1 = random.choice(candidates)
                if old_left_foot:
                    note0, note1 = note1, note0

                new_state_pairs.append((
                    ((note0, note1), old_left_foot),
                    (old_score, (note1, (note0, old_full_steps))),
                ))
            elif note_kind == JUMP:
                if old_full_steps is None:
                    if old_left_foot:
                        note0 = 4
                        note1 = 3
                    else:
                        note0 = 3
                        note1 = 4
                else:
                    note1 = old_full_steps[0]
                    
                    if old_left_foot:
                        candidates = [
                            b for a, b in jumps_for_position[positions[position_index]]
                            if a == note1
                        ]
                    else:
                        candidates = [
                            a for a, b in jumps_for_position[positions[position_index]]
                            if b == note1
                        ]

                    note0 = random.choice(candidates)

                new_state_pairs.append((
                    ((note0, note1), old_left_foot),
                    (old_score, (note1, (note0, old_full_steps))),
                ))
            elif note_kind == JUMP_JACK:
                note1 = old_full_steps[0]
                note0 = old_full_steps[1][0]

                new_state_pairs.append((
                    ((note0, note1), old_left_foot),
                    (old_score, (note1, (note0, old_full_steps))),
                ))
            elif was_jump:
                assert note_kind in [SLIDE, STEP], f'wtf? {note_kind}'
                old1 = old_full_steps[0]
                old0 = old_full_steps[1][0]
                new_state_pairs.append((
                    ((old0, old1, old0), not old_left_foot),
                    (old_score, (old0, old_full_steps)),
                ))
            else:
                assert note_kind in [STEP, SLIDE], 'wtf2?'
                for new_step in range(8):
                    new_steps = (old_steps + (new_step,))[-STATE_STEP_COUNT:]
                    new_left_foot = not old_left_foot
                    new_full_steps = (new_step, old_full_steps)
                    new_score = old_score + rate_step(
                        new_steps,
                        new_left_foot,
                        position_index,
                        note_kind == STEP,
                        og_note,
                    )
                    if new_score != NEVER:
                        new_state_pairs.append((
                            (new_steps, new_left_foot),
                            (new_score, new_full_steps),
                        ))

        was_jump = note_kind in [JUMP, JUMP_JACK]
        if was_jump:
            # Don't transition immediately after a jump.
            position_progress = min(0.999, position_progress)

        # Sort by score decreasing, breaking ties randomly.
        random.shuffle(new_state_pairs)
        new_state_pairs.sort(key=lambda pair: -pair[1][0])

        # Select the top BEAM_SIZE states.
        new_state_pairs = new_state_pairs[:BEAM_SIZE]

        states = {}
        for key, value in new_state_pairs:
            score, _ = value
            if key in states:
                existing_score = states[key][0]
                if existing_score >= score:
                    continue

            states[key] = value

    if len(states) == 0:
        return None


    best_value = max(states.values(), key=lambda value: value[0])
    score, full_steps = best_value
    double_steps = []
    while full_steps is not None:
        double_steps.append(full_steps[0])
        full_steps = full_steps[1]
    double_steps = double_steps[::-1]
    
    double_step_index = 0
    held_notes = set()
    cancelled_hold_count = 0
    double_notes = []
    for measure in measures:
        for notes in measure:
            chars = [b'0']*8
            new_held_notes = set()
            
            tap_count = sum(n in b'1' for n in notes)
            hold_count = sum(n in b'2' for n in notes)
            roll_count = sum(n in b'4' for n in notes)

            while tap_count + roll_count + hold_count > 2:
                if tap_count > 0:
                    tap_count -= 1
                elif hold_count > 0:
                    hold_count -= 1
                    cancelled_hold_count += 1
                else:
                    roll_count -= 1
                    cancelled_hold_count += 1

            for kind, count in zip(b'124', [tap_count, hold_count, roll_count]):
                for _ in range(count):
                    n = double_steps[double_step_index]
                    double_step_index += 1
                    chars[n] = bytes([kind])

                    if n in held_notes:
                        # Cancel the hold.
                        if double_notes[-1][n] in b'24':
                            cancel_note = b'1'
                        else:
                            cancel_note = b'3'
                        double_notes[-1] = double_notes[-1][:n] + cancel_note + double_notes[-1][n+1:]
                        held_notes.remove(n)
                        cancelled_hold_count += 1

                    if kind in b'24':
                        new_held_notes.add(n)

            release_count = sum(n in b'3' for n in notes)
            for _ in range(release_count):
                if cancelled_hold_count > 0:
                    cancelled_hold_count -= 1
                elif len(held_notes) > 0:
                    n = next(iter(held_notes))
                    chars[n] = b'3'
                    held_notes.remove(n)
                else:
                    log.warning(f'Something\'s fucky with the holds?')

            held_notes |= new_held_notes

            double_notes.append(b''.join(chars))

    assert double_step_index == len(double_steps)

    double_measures = []
    double_notes_index = 0
    for measure in measures:
        double_measures.append([])
        for _ in measure:
            double_measures[-1].append(double_notes[double_notes_index])
            double_notes_index += 1

    for measure_index in range(len(measures)):
        single_measure = measures[measure_index]
        double_measure = double_measures[measure_index]

        for notes_index in range(len(single_measure)):
            mine_count = sum(k in b'M' for k in single_measure[notes_index])

            if mine_count == 0:
                continue

            candidates = set(range(8))
            
            for measure_offset in range(-1, 2):
                i = measure_index + measure_offset
                if i < 0:
                    continue
                if i >= len(measures):
                    continue
                for notes in double_measures[i]:
                    for n, k in enumerate(notes):
                        if k in b'1234':
                            candidates.discard(n)

            mine_count = min(mine_count, len(candidates))
            for n in random.sample(list(candidates), mine_count):
                double_measure[notes_index] = double_measure[notes_index][:n] + b'M' + double_measure[notes_index][n+1:]

    return double_measures


def normalize_notes(notes):
    if b'//' in notes:
        notes = notes[:notes.index(b'//')]

    return notes.strip()

def doublify_notes_data(notes_data, bpm_changes):
    # Seed RNG with notes data for consistent charts.
    random.seed(notes_data)

    measures = [
        [normalize_notes(notes) for notes in measure.split(b'\n')]
        for measure in notes_data.split(b',')
    ]

    measures = [
        [notes for notes in measure if notes]
        for measure in measures
    ]

    double_measures = None
    retry_count = 10
    for i in range(retry_count):
        if i != 0:
            log.warning(f'Failed attempt to doublify. Retry #{i}.')
        double_measures = doublify_measures(measures, bpm_changes)
        if double_measures is not None:
            break
    if double_measures is None:
        raise Exception('Failed to doublify after 10 tries!')

    double_measures = b'\r\n,\r\n'.join(
        b'\r\n'.join(measure)
        for measure in double_measures
    ) + b'\r\n'

    return double_measures

def has_non_autogen_double_sm(pairs):
    for key, value in pairs:
        if key != b'NOTES': continue
        lines = value.split(b'\n')
        if len(lines) >= 3 and lines[1].strip() == b'dance-double:' and not lines[2].startswith(b'     AUTO '):
            return True

def is_double_chart_sm(key, value):
    if key != b'NOTES': return False
    lines = value.split(b'\n')
    return len(lines) >= 3 and lines[1].strip() == b'dance-double:'

def parse_bpms(raw_bpms):
    if len(raw_bpms) == 0:
        log.warning('No bpms???')
        return

    bpm_changes = [
        tuple(float(x.strip()) for x in part.split(b'='))
        for part in raw_bpms.split(b',')
    ]
    
    return bpm_changes

def doublify_sm(data):
    sm = parse_sm(data)

    # Skip files which have non-autogen double charts.
    if has_non_autogen_double_sm(sm):
        log.info('SKIPPED')
        return None

    # Remove double charts.
    sm = [
        pair for pair in sm
        if not is_double_chart_sm(*pair)
    ]

    for key, value in sm:
        if key == b'BPMS':
            bpm_changes = parse_bpms(value)
            break

    for key, value in sm[:]:
        if key != b'NOTES': continue
        try:
            lines = value.split(b'\n')
            header = lines[:6]
            notes_data = b'\n'.join(lines[6:])

            if b'dance-single:' not in header[1]:
                continue

            header[1] = b'     dance-double:\r'
            difficulty = header[3]
            log.info(f'  - {difficulty}')

            header[2] = b'     AUTO ' + header[2].strip()

            notes_data = doublify_notes_data(notes_data, bpm_changes)

            value = b'\n'.join(header) + b'\n' + notes_data
            sm.append((b'NOTES', value))
        except Exception:
            log.exception(f'Failed to doublify {difficulty}')

    return encode_sm(sm)

SSC_DESCRIPTION_KEYS = [b'CHARTNAME', b'DESCRIPTION', b'CREDIT']

def has_non_autogen_double_ssc(charts):
    chart_values = {}

    for chart in charts:
        chart_values = {key: value for key, value in chart if key is not None}
        is_double = chart_values.get(b'STEPSTYPE') == b'dance-double'
        is_auto = all(
            chart_values.get(key, b'').startswith(b'AUTO')
            for key in SSC_DESCRIPTION_KEYS
        )
        if is_double and not is_auto:
            return True

def split_ssc(ssc):
    song = []
    charts = []

    cur_chart = None
    for key, value in ssc:
        if key == b'NOTEDATA':
            cur_chart = []
            charts.append(cur_chart)

        if cur_chart is None:
            song.append((key, value))
        else:
            cur_chart.append((key, value))

    return song, charts

def doublify_ssc(data):
    ssc = parse_sm(data)

    song, charts = split_ssc(ssc)

    # Skip files which have non-autogen double charts.
    if has_non_autogen_double_ssc(charts):
        log.info('SKIPPED')
        return

    # Remove double charts.
    charts = [
        chart for chart in charts
        if (b'STEPSTYPE', b'dance-double') not in chart
    ]

    song_bpms = None
    for key, value in song:
        if key == b'BPMS':
            song_bpms = value
            break

    for chart in charts[:]:
        try:
            difficulty = None
            chart_values = {key: value for key, value in chart if key is not None}
            if chart_values.get(b'STEPSTYPE') != b'dance-single':
                continue

            notes_data = chart_values.get(b'NOTES')
            if notes_data is None:
                continue

            for key in SSC_DESCRIPTION_KEYS:
                chart_values[key] = b'AUTOv1.0 ' + chart_values.get(key, b'')

            chart_values[b'STEPSTYPE'] = b'dance-double'

            difficulty = chart_values.get(b'DIFFICULTY', b'???')
            log.info(f'  - {difficulty}')

            bpm_changes = parse_bpms(chart_values.get(b'BPMS', song_bpms))

            notes_data = doublify_notes_data(notes_data, bpm_changes)

            chart_values[b'NOTES'] = b'\n' + notes_data

            double_chart = []
            for key, value in chart:
                if key is None:
                    double_chart.append((key, value))
                else:
                    double_chart.append((key, chart_values[key]))
            charts.append(double_chart)
        except Exception:
            log.exception(f'Failed to doublify {difficulty}')

    ssc = song[:]
    for chart in charts:
        ssc += chart

    return encode_sm(ssc)

def doublify_path(path):
    if os.path.isdir(path):
        for filename in os.listdir(path):
            doublify_path(os.path.join(path, filename))
    else:
        lower_path = path.lower()
        is_sm = lower_path.endswith('.sm')
        is_ssc = lower_path.endswith('.ssc')
        if is_sm or is_ssc:
            log.info(f'Doublifying {path}')
            with open(path, 'rb') as file:
                data = file.read()
            try:
                if is_ssc:
                    data = doublify_ssc(data)
                else:
                    data = doublify_sm(data)
            except Exception:
                log.exception(f'Something went wrong doublifying {path}')
                return
            if data is None:
                return
            with open(path, 'wb') as file:
                file.write(data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        log.error('Please supply a path to a folder or .sm file.')
        sys.exit(1)

    for path in sys.argv[1:]:
        doublify_path(path)
