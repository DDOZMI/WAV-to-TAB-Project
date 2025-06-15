import os
import sys
import math
import json
import itertools
import traceback
from enum import Enum
from pathlib import Path
from collections import defaultdict
from time import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import pretty_midi
from pretty_midi.containers import TimeSignature
from pretty_midi import note_number_to_name, note_name_to_number


class Note:
    def __init__(self, pitch: int):
        self.pitch = pitch
        self.name = note_number_to_name(pitch)
        self.degree = self.name[:-1]
        self.octave = self.name[-1]

    def __eq__(self, other):
        return isinstance(other, Note) and self.pitch == other.pitch
    
    def __hash__(self):
        return hash((self.name, id(self)))

    def __repr__(self):
        return self.name + "-" + str(hash(self))[2:4]


class Degree(Enum):
    A = "A"
    Asharp = "A#"
    B = "B"
    C = "C"
    Csharp = "C#"
    D = "D"
    Dsharp = "D#"
    E = "E"
    F = "F"
    Fsharp = "F#"
    G = "G"
    Gsharp = "G#"


class Tuning:
    standard_tuning = ["E4", "B3", "G3", "D3", "A2", "E2"]
    standard_ukulele_tuning = ["A4", "E4", "C4", "G4"]

    def __init__(self, strings=None):
        if strings is None:
            strings = self.standard_tuning
        self._strings = np.array([Note(note_name_to_number(note)) for note in strings])
        self.nstrings = len(strings)
        self.nfrets = 20

    @property
    def strings(self):
        return self._strings
    
    def get_all_possible_notes(self):
        """프렛 상에서 가능한 모든 음표 반환"""
        res = []
        for string in self.strings:
            string_notes = []
            for ifret in range(self.nfrets + 1):
                string_notes.append(Note(string.pitch + ifret))
            res.append(string_notes)
        return res
    
    def get_pitch_bounds(self):
        """연주 가능한 음높이 범위 반환"""
        min_pitch = min([string.pitch for string in self.strings])
        max_pitch = max([string.pitch for string in self.strings]) + self.nfrets
        return min_pitch, max_pitch

    def get_bounds(self):
        """튜닝의 경계 음표들 반환"""
        min_pitch, max_pitch = self.get_pitch_bounds()
        return [Note(min_pitch), Note(max_pitch)]


def measure_length_ticks(midi, time_signature): 
    """한 마디의 틱 수 계산"""
    n_quarter_notes = time_signature.numerator * (4/time_signature.denominator)
    measure_length = n_quarter_notes * midi.resolution 
    return measure_length


def get_notes_between(midi, notes, begin, end):
    """특정 시간 범위의 음표 반환"""
    res = []
    for note in notes:
        note_start_ticks = midi.time_to_tick(note.start)
        if note_start_ticks >= begin and note_start_ticks < end:
            res.append(note)
    return res


def get_non_drum(instruments):
    """드럼이 아닌 악기들만 반환"""
    return [instrument for instrument in instruments if not instrument.is_drum]


def fill_measure_str(str_array):
    """문자열 배열을 같은 길이로 맞추기"""
    maxlen = len(max(str_array, key=len))
    res = []
    for s in str_array:
        res.append(s.ljust(maxlen, "-"))
    return res


def sort_notes_by_tick(notes):
    """틱 순서로 음표 정렬"""
    return sorted(notes, key=lambda n: n.start)


def round_to_multiple(n, base=10):
    """지정된 배수로 반올림"""
    return int(base * round(n/base))


def quantize(midi):
    """MIDI 정렬"""
    quantization_factor = 32
    
    for instrument in midi.instruments:
        quantized_notes = []
        for note in instrument.notes:
            rounded = round_to_multiple(midi.time_to_tick(note.start), 
                                      base=midi.resolution/quantization_factor)
            quantized_notes.append(pretty_midi.Note(
                velocity=note.velocity, 
                pitch=note.pitch, 
                start=midi.tick_to_time(rounded), 
                end=note.end))
        instrument.notes = quantized_notes


def transpose_note(note, semitones):
    """음표를 반음 단위로 이조"""
    return Note(note.pitch + semitones)


def remove_duplicate_notes(notes):
    """중복 음표 제거"""
    pitches = []
    res_notes = []
    for note in notes:
        if note.pitch not in pitches:
            pitches.append(note.pitch)
            res_notes.append(note)
    return res_notes


def sort_notes_by_pitch(notes):
    """음높이 순으로 정렬"""
    return sorted(notes, key=lambda n: n.pitch)


def get_events_between(timeline, start_ticks, end_ticks):
    """특정 시간 범위의 이벤트들 반환"""
    return {key: timeline[key] for key in timeline.keys() 
            if start_ticks <= key < end_ticks}
    

def build_path_graph(G, note_arrays):
    """가능한 모든 음표 조합의 경로 그래프 생성"""
    res = nx.DiGraph()

    for x, note_array in enumerate(note_arrays):
        for y, possible_note in enumerate(note_array):
            res.add_node(possible_note, pos=(x, y))

    for idx, note_array in enumerate(note_arrays[:-1]):
        for possible_note in note_array:
            for possible_target_note in note_arrays[idx+1]:
                distance = G[possible_note][possible_target_note]["distance"]
                if is_edge_possible(possible_note, possible_target_note, G):
                    res.add_edge(possible_note, possible_target_note, distance=distance)

    return res 


def is_edge_possible(possible_note, possible_target_note, G):
    """두 노드 간 연결 가능성 확인"""
    is_distance_possible = G[possible_note][possible_target_note]["distance"] < 6
    is_different_string = (G.nodes[possible_note]["pos"][0] != 
                          G.nodes[possible_target_note]["pos"][0])
    return is_distance_possible and is_different_string


def is_path_already_checked(paths, current_path):
    """경로가 이미 확인되었는지 검사"""
    for path in paths:
        if set(path) == set(current_path):
            return True
    return False


def laplace_distro(x, b, mu=0.0):
    """라플라스 분포 계산"""
    return (1/(2*b))*math.exp(-abs(x-mu)/(b))


def get_nfingers(G, path):
    """필요한 손가락 수 계산"""
    count = 0
    for note in path:
        ifret = G.nodes[note]["pos"][1]
        if ifret != 0:
            count += 1
    return count


def get_n_changed_strings(G, path, previous_path, tuning):
    """이전 형태에서 바뀐 현의 수 계산"""
    used_strings = set([G.nodes[note]["pos"][0] for note in path])
    previous_used_strings = set([G.nodes[note]["pos"][0] for note in previous_path 
                               if G.nodes[note]["pos"][1] != 0])
    
    n_changed_strings = len(path) - len(set(used_strings).intersection(previous_used_strings))
    n_changed_strings_score = n_changed_strings/tuning.nstrings
    assert 0 <= n_changed_strings_score <= 1
    return n_changed_strings_score


def get_height_score(G, path, tuning, previous_path=None):
    """높이 점수 계산 (0-1)"""
    height = get_raw_height(G, path, previous_path)/tuning.nfrets
    assert 0 <= height <= 1
    return height


def get_raw_height(G, path, previous_path=None):
    """프렛보드에서의 평균 높이 계산"""
    y = [G.nodes[note]["pos"][1] for note in path if G.nodes[note]["pos"][1] != 0]

    if len(y) > 0:
        return (max(y) + min(y))/2
    elif previous_path is None:
        return 0
    else:
        return get_raw_height(G, previous_path)


def get_dheight_score(height, previous_height, tuning):
    """이전 핑거링과의 높이 차이 점수"""
    dheight = np.abs(height-previous_height)/tuning.nfrets
    assert 0 <= dheight <= 1
    return dheight


def get_path_span(G, path):
    """경로의 수직 범위 계산"""
    y = [G.nodes[note]["pos"][1] for note in path if G.nodes[note]["pos"][1] != 0]
    
    span = (max(y) - min(y))/5 if len(y) > 0 else 0
    assert 0 <= span <= 1
    return span


def compute_path_difficulty(G, path, previous_path, weights, tuning):
    """경로 난이도 계산"""
    raw_height = get_raw_height(G, path, previous_path)
    previous_raw_height = get_raw_height(G, previous_path) if len(previous_path) > 0 else 0
    
    height = get_height_score(G, path, tuning, previous_path)
    dheight = get_dheight_score(raw_height, previous_raw_height, tuning)
    span = get_path_span(G, path)
    n_changed_strings = get_n_changed_strings(G, path, previous_path, tuning)
    
    easiness = (laplace_distro(dheight, b=weights["b"]) * 
               1/(1+height * weights["height"]) * 
               1/(1+span * weights["length"]) * 
               1/(1+n_changed_strings * weights["n_changed_strings"]))
    
    return 1/easiness


def compute_isolated_path_difficulty(G, path, tuning):
    """독립 경로 난이도 계산"""
    height = get_height_score(G, path, tuning)
    span = get_path_span(G, path)
    
    easiness = 1/(1+height) * 1/(1+span)
    return 1/easiness


def safe_log(x, min_val=1e-10):
    """로그 계산"""
    return np.log(np.maximum(x, min_val))


def viterbi(V, Tm, Em, initial_distribution=None):
    """비터비 알고리즘 구현"""
    T = len(V)
    M = Tm.shape[0]

    if initial_distribution is None:
        initial_distribution = np.full(M, 1/M)

    omega = np.zeros((T, M))
    # 로그 계산으로 경고 방지
    omega[0, :] = safe_log(initial_distribution * Em[:, V[0]])

    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            # 로그 계산으로 경고 방지
            probability = omega[t - 1] + safe_log(Tm[:, j]) + safe_log(Em[j, V[t]])
            prev[t - 1, j] = np.argmax(probability)
            omega[t, j] = np.max(probability)

    S = np.zeros(T)
    last_state = np.argmax(omega[T - 1, :])
    S[0] = last_state

    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    S = np.flip(S, axis=0)
    return S.astype(int)


def build_transition_matrix(G, fingerings, weights, tuning):
    """전이 행렬 구축"""
    transition_matrix = np.zeros((len(fingerings), len(fingerings)))
    for iprevious in range(len(fingerings)):
        difficulties = np.array([1/compute_path_difficulty(G, fingerings[icurrent], 
                                                         fingerings[iprevious], weights, tuning)
                               for icurrent in range(len(fingerings))])
        
        transition_matrix[iprevious] = difficulties_to_probabilities(difficulties)
    
    return transition_matrix


def difficulties_to_probabilities(difficulties):
    """난이도를 확률로 변환"""
    difficulties_total = np.sum(difficulties)
    return np.array([difficulty/difficulties_total for difficulty in difficulties])


def expand_emission_matrix(emission_matrix, all_paths):
    """방출 행렬"""
    if len(emission_matrix) > 0:
        filler = np.zeros((len(all_paths), emission_matrix.shape[1]))
        emission_matrix = np.vstack((emission_matrix, filler))
        column = np.vstack((np.zeros((len(emission_matrix)-len(all_paths), 1)), 
                           np.ones((len(all_paths), 1))))
        emission_matrix = np.hstack((emission_matrix, column))
    else:
        emission_matrix = np.ones((len(all_paths), 1))

    return emission_matrix


class Fretboard:
    def __init__(self, tuning):
        self.tuning = tuning
        self.nstrings = tuning.nstrings
        self.scale_length = 650
        self.G = self._build_complete_graph()

    def _build_complete_graph(self):
        """프렛보드를 나타내는 완전 그래프 구축"""
        note_map = self.tuning.get_all_possible_notes()

        complete_graph = nx.Graph()
        for istring, string in enumerate(note_map):
            for inote, note in enumerate(string):
                complete_graph.add_node(note, pos=(istring, inote))

        complete_graph_nodes = list(complete_graph.nodes(data=True))

        for node in complete_graph_nodes:
            for node_to_link in complete_graph_nodes:
                dst = (0 if node_to_link[1]['pos'][1] == 0 
                      else self.distance_between(node[1]['pos'], node_to_link[1]['pos']))
                complete_graph.add_edge(node[0], node_to_link[0], distance=dst)
            
        return complete_graph
    
    def get_note_options(self, notes):
        """음표 리스트에 대한 선택사항 반환"""
        note_options = [self.get_specific_note_options(note) for note in notes]
        note_options = [note_array for note_array in note_options if len(note_array) > 0]
        return note_options
    
    def get_specific_note_options(self, note):
        """특정 음표에 대한 모든 노드 반환"""
        nodes = list(self.G.nodes)
        return [node for node in nodes if node == note]
    
    def get_possible_fingerings(self, note_options): 
        """가능한 모든 핑거링 반환"""
        fingerings = []
        
        if len(note_options) == 1:
            return [(note,) for note in note_options[0]]

        for note_options_permutation in list(itertools.permutations(note_options)):
            path_graph = build_path_graph(self.G, note_options_permutation)
            for possible_source_node in note_options_permutation[0]:
                permutation_fingerings = nx.all_simple_paths(
                    path_graph, possible_source_node, target=note_options_permutation[-1])
                for fingering in permutation_fingerings:
                    if (not is_path_already_checked(fingerings, fingering) and 
                        self.is_fingering_possible(fingering, note_options_permutation)):
                        fingerings.append(tuple(fingering))
                    
        return fingerings
    
    def fix_oob_notes(self, notes, preserve_highest_note=False):
        """범위를 벗어난 음표들 수정"""
        min_possible_pitch, max_possible_pitch = self.tuning.get_pitch_bounds()
        
        if preserve_highest_note:
            highest_pitch_before = max([note.pitch for note in notes])
            
            if highest_pitch_before > max_possible_pitch:
                highest_pitch_semitones_above_max = max(highest_pitch_before - max_possible_pitch, 0)
                highest_pitch_after = highest_pitch_before - (math.ceil(highest_pitch_semitones_above_max/12) * 12)
            else:
                highest_pitch_semitones_below_max = max(min_possible_pitch - highest_pitch_before, 0)
                highest_pitch_after = highest_pitch_before + (math.ceil(highest_pitch_semitones_below_max/12) * 12)
            
            max_possible_pitch = highest_pitch_after
        
        res_notes = []
        
        for note in notes:
            n_octaves_to_adjust = 0
            
            if note.pitch > max_possible_pitch:
                semitones_above_max = max(note.pitch - max_possible_pitch, 0)
                n_octaves_to_adjust = -math.ceil(semitones_above_max/12)
            
            if note.pitch < min_possible_pitch:
                semitones_below_min = max(min_possible_pitch - note.pitch, 0)
                n_octaves_to_adjust = math.ceil(semitones_below_min/12)
            
            new_note = transpose_note(note, n_octaves_to_adjust * 12)
            
            if new_note.pitch >= min_possible_pitch and new_note.pitch <= max_possible_pitch:
                res_notes.append(new_note)
            
        return remove_duplicate_notes(res_notes)  
    
    def distance_between(self, p1, p2):
        """프렛보드에서 두 점 사이의 거리 계산"""
        p1 = (p1[0]/self.nstrings, p1[1])
        p2 = (p2[0]/self.nstrings, p2[1])
        return math.dist(p1, p2)
    
    def is_fingering_possible(self, fingering, note_arrays):
        """핑거링이 가능하고 연주 가능한지 확인"""
        # 한 현에 하나의 손가락만
        plucked_strings = [self.G.nodes[note]["pos"][0] for note in fingering]
        one_per_string = len(plucked_strings) == len(set(plucked_strings))

        # 5프렛 이내 범위
        used_frets = [self.G.nodes[note]["pos"][1] for note in fingering 
                     if self.G.nodes[note]["pos"][1] != 0]
        max_fret_span = (max(used_frets) - min(used_frets)) < 5 if len(used_frets) > 0 else True

        # 필요한 것보다 더 많은 노드를 방문하지 않음
        right_length = len(fingering) <= len(note_arrays)

        return one_per_string and max_fret_span and right_length


# ============================================================================
# 배치 클래스
# ============================================================================

class Arrangement:
    """음표 배치 클래스"""
    def __init__(self, notes, tuning):
        self.notes = notes
        self.tuning = tuning
        
    def fit_notes_to_tuning(self):
        """음표들을 튜닝에 맞게 조정"""
        pitch_bounds = [note.pitch for note in self.tuning.get_bounds()]
        
        notes_pitches = [note.pitch for note in self.notes]
        
        for i in range(len(notes_pitches)):
            notes_pitches[i] = self.fit_note_to_tuning(notes_pitches[i], pitch_bounds)
                
        self.notes = [Note(pitch) for pitch in notes_pitches]
        
    def fit_note_to_tuning(self, pitch, pitch_bounds):
        """단일 음표를 튜닝에 맞게 조정"""
        while pitch < pitch_bounds[0]:
            pitch += 12
                
        while pitch > pitch_bounds[1]:
            pitch -= 12
        
        return pitch


class Measure: 
    """마디 클래스"""
    def __init__(self, tab, imeasure, time_signature, measure_start, measure_end):
        self.imeasure = imeasure
        self.time_signature = time_signature
        self.tab = tab
        
        self.measure_start = measure_start
        self.measure_end = measure_end
        
        self.timeline = get_events_between(self.tab.timeline, measure_start, measure_end)

    @property
    def duration_ticks(self):
        return self.measure_end - self.measure_start


class Tab:
    """탭 변환 메인 클래스"""
    def __init__(self, name, tuning, midi, output_dir=None, weights=None):
        self.name = name
        self.tuning = tuning
        self.time_signatures = (midi.time_signature_changes if len(midi.time_signature_changes) > 0 
                               else [TimeSignature(4, 4, 0)])
        self.nstrings = len(tuning.strings)
        self.measures = []
        self.midi = midi
        self.fretboard = Fretboard(tuning)
        self.weights = ({"b":1, "height":1, "length":1, "n_changed_strings":1} 
                       if weights is None else weights)
        self.timeline = self.build_timeline()
        self.output_dir = output_dir
        
        self.populate()
        self.tab = self.gen_tab()

    def populate(self):
        """탭을 마디로 채우기"""
        for i, time_signature in enumerate(self.time_signatures):
            measure_length_in_ticks = measure_length_ticks(self.midi, time_signature)
            
            time_sig_start = self.midi.time_to_tick(time_signature.time)
            
            time_sig_end = (self.time_signatures[i+1].time if i < len(self.time_signatures)-1 
                           else self.midi.get_end_time())
            time_sig_end = self.midi.time_to_tick(time_sig_end)
            
            measure_ticks = np.arange(time_sig_start, time_sig_end, measure_length_in_ticks)
            
            for imeasure, measure_start in enumerate(measure_ticks):
                measure_end = min(measure_start + measure_length_in_ticks, time_sig_end)
                self.measures.append(Measure(self, imeasure, time_signature, measure_start, measure_end))
    
    def build_timeline(self):
        """타임라인 구축"""
        timeline = defaultdict(dict)
        non_drum_instruments = get_non_drum(self.midi.instruments)
        
        # 음표들
        for instrument in non_drum_instruments:
            notes = instrument.notes
            notes.sort(key=lambda x: x.start)
            
            for note in notes:
                note_tick = self.midi.time_to_tick(note.start)
                
                if "notes" in timeline[note_tick]:
                    timeline[note_tick]["notes"].append(note)
                else:
                    timeline[note_tick]["notes"] = [note]
        
        # 박자표
        for time_signature in self.time_signatures:
            time_signature_tick = self.midi.time_to_tick(time_signature.time)
            timeline[time_signature_tick]["time_signature"] = time_signature
            
        return timeline

    def gen_tab(self):
        tab = {}
        tab["tuning"] = [string.pitch for string in self.tuning.strings]
        tab["measures"] = []

        notes_vocabulary = []
        notes_sequence = []
        fingerings_vocabulary = []
        emission_matrix = np.array([])
        initial_probabilities = None

        for measure in self.measures:
            res_measure = {"events": []}
            measure_events = measure.timeline

            for event_tick, event_types in measure_events.items():        
                event = {
                    "time": self.midi.tick_to_time(int(event_tick)),
                    "time_ticks": int(event_tick),
                    "measure_timing": (event_tick - measure.measure_start)/measure.duration_ticks
                }
                
                if "time_signature" in event_types:
                    ts = event_types["time_signature"]
                    event["time_signature_change"] = [ts.numerator, ts.denominator]

                if "notes" in event_types:
                    event["notes"] = []
                    notes = event_types["notes"]
                    notes_pitches = tuple(set([note.pitch for note in notes]))
                    notes = [Note(pitch) for pitch in notes_pitches]
                    notes = self.fretboard.fix_oob_notes(notes, preserve_highest_note=False)
                    note_options = self.fretboard.get_note_options(notes)
                    
                    if notes_pitches not in notes_vocabulary:
                        fingering_options = self.fretboard.get_possible_fingerings(note_options)
                        
                        if len(fingering_options) > 0:   
                            notes_vocabulary.append(notes_pitches)
                            fingerings_vocabulary += fingering_options 
                                        
                            if initial_probabilities is None:
                                isolated_difficulties = [compute_isolated_path_difficulty(
                                    self.fretboard.G, path, self.tuning) for path in fingering_options]
                                initial_probabilities = difficulties_to_probabilities(isolated_difficulties)
                            
                            emission_matrix = expand_emission_matrix(emission_matrix, fingering_options)
                        
                    if notes_pitches in notes_vocabulary:
                        notes_sequence.append(notes_vocabulary.index(notes_pitches))
                    else:
                        notes_sequence.append(-1)
                        
                res_measure["events"].append(event)
            
            tab["measures"].append(res_measure)
                
        # 비터비 알고리즘으로 최적 시퀀스 찾기
        if len(fingerings_vocabulary) > 0 and len(notes_sequence) > 0:
            transition_matrix = build_transition_matrix(
                self.fretboard.G, fingerings_vocabulary, self.weights, self.tuning)
            
            initial_probabilities = np.hstack((
                initial_probabilities, 
                np.zeros(len(transition_matrix) - len(initial_probabilities))))
            
            valid_sequence = [i for i in notes_sequence if i >= 0]
            if len(valid_sequence) > 0:
                sequence_indices = viterbi(valid_sequence, transition_matrix, 
                                         emission_matrix, initial_probabilities)
                final_sequence = np.array(fingerings_vocabulary, dtype=object)[sequence_indices]
                tab = self.populate_tab_notes(tab, final_sequence)
        
        return tab

    def populate_tab_notes(self, tab, sequence):
        """탭 템플릿에 음표와 핑거링 채우기"""
        ievent = 0
        for measure in tab["measures"]:
            for event in measure["events"]:
                if "notes" not in event:
                    continue
                
                if ievent < len(sequence):
                    for path_note in sequence[ievent]:
                        string, fret = self.fretboard.G.nodes[path_note]["pos"]   
                        event["notes"].append({
                            "degree": path_note.degree,
                            "octave": path_note.octave,
                            "string": string,
                            "fret": fret
                        })
                    ievent += 1
                    
        return tab

    def to_string(self):
        """ASCII 탭 텍스트 생성"""
        res = []
        for string in self.tuning.strings:
            header = string.degree
            header += "||" if len(header) > 1 else " ||"
            res.append(header)

        for measure in self.tab["measures"]:
            for ievent, event in enumerate(measure["events"]):
                if "notes" in event and len(event["notes"]) > 0:
                    # 각 현에 대해 프렛 번호 추가
                    string_frets = {}
                    for note in event["notes"]:
                        string, fret = note["string"], note["fret"]
                        string_frets[string] = str(fret)
                    
                    # 모든 현에 대해 프렛 번호 또는 대시 추가
                    for istring in range(self.nstrings):
                        if istring in string_frets:
                            res[istring] += string_frets[istring]
                        else:
                            res[istring] += "-"

                    # 다음 이벤트까지의 간격 계산
                    next_event_timing = (measure["events"][ievent + 1]["measure_timing"] 
                                       if ievent < len(measure["events"]) - 1 else 1.0)
                    dashes_to_add = max(1, math.floor((next_event_timing - event["measure_timing"]) * 16))

                    # 문자열 길이 맞추기
                    res = fill_measure_str(res)

                    # 대시 추가
                    for istring in range(self.nstrings):
                        res[istring] += "-" * dashes_to_add

            # 마디 구분선 추가
            for istring in range(self.nstrings):
                res[istring] += "|"

        return res

    def save_ascii(self, filename=None):
        """ASCII 탭을 파일로 저장"""
        if filename is None:
            filename = f"{self.name}.txt"
        
        notes_str = self.to_string()
        
        with open(filename, "w", encoding="utf-8") as file:
            file.write(f"Title: {self.name}\n")
            file.write("=" * 50 + "\n\n")
            for string_notes in notes_str:
                file.write(string_notes + "\n")
        
        print(f"TAB 파일이 저장되었습니다: {filename}")


class MidiToTabConverter:
    """MIDI를 TAB으로 변환하는 메인 클래스"""
    
    def __init__(self, tuning=None, weights=None):
        """
        Args:
            tuning: 악기 튜닝 (None이면 표준 기타 튜닝 사용)
            weights: 난이도 계산 가중치
        """
        self.tuning = Tuning(tuning) if tuning else Tuning()
        self.weights = weights or {"b": 1, "height": 1, "length": 1, "n_changed_strings": 1}
    
    def convert_file(self, midi_path, output_path=None, song_name=None):
        """
        MIDI 파일을 TAB으로 변환
        
        Args:
            midi_path: 입력 MIDI 파일 경로
            output_path: 출력 TAB 파일 경로 (None이면 자동 생성)
            song_name: 곡 제목 (None이면 파일명 사용)
        """
        try:
            # MIDI 파일 로드
            print(f"MIDI 파일 로딩 중: {midi_path}")
            midi = pretty_midi.PrettyMIDI(midi_path)
            
            # 곡 제목 설정
            if song_name is None:
                song_name = Path(midi_path).stem
            
            # 출력 파일명 설정
            if output_path is None:
                output_path = f"{song_name}_tab.txt"
            
            print(f"변환 중... (곡: {song_name})")
            
            # TAB 생성
            tab = Tab(song_name, self.tuning, midi, weights=self.weights)
            
            # ASCII TAB 저장
            tab.save_ascii(output_path)
            
            return tab
            
        except Exception as e:
            print(f"변환 중 오류 발생: {e}")
            traceback.print_exc()
            return None
    
    def convert_and_display(self, midi_path, song_name=None):
        """
        MIDI 파일을 변환하고 콘솔에 출력
        
        Args:
            midi_path: 입력 MIDI 파일 경로
            song_name: 곡 제목
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            
            if song_name is None:
                song_name = Path(midi_path).stem
            
            print(f"=== {song_name} ===")
            
            tab = Tab(song_name, self.tuning, midi, weights=self.weights)
            tab_strings = tab.to_string()
            
            for line in tab_strings:
                print(line)
                
            return tab
            
        except Exception as e:
            print(f"변환 중 오류 발생: {e}")
            traceback.print_exc()
            return None


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python midi_to_tab.py <midi_file> [output_file] [song_name]")
        print("")
        print("예시:")
        print("  python midi_to_tab.py song.mid")
        print("  python midi_to_tab.py song.mid output.txt")
        print("  python midi_to_tab.py song.mid output.txt \"My Song\"")
        return
    
    midi_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    song_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(midi_path):
        print(f"MIDI 파일을 찾을 수 없습니다: {midi_path}")
        return
    
    # 변환기 생성 및 실행
    converter = MidiToTabConverter()
    tab = converter.convert_file(midi_path, output_path, song_name)
    
    if tab:
        print("변환이 완료되었습니다!")
    else:
        print("변환에 실패했습니다.")


if __name__ == "__main__":
    main()