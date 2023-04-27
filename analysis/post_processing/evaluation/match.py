import numpy as np
from collections import OrderedDict

from analysis.post_processing import post_processing
from mlreco.utils.globals import *
from analysis.classes.matching import (match_particles_fn,
                                       match_particles_optimal,
                                       match_interactions_fn,
                                       match_interactions_optimal)
from analysis.classes.data import *

@post_processing(data_capture=['index'], 
                 result_capture=['particles',
                                 'truth_particles'])
def match_particles(data_dict,
                    result_dict,
                    matching_mode='optimal',
                    matching_direction='pred_to_true',
                    match_particles=True,
                    min_overlap=0,
                    overlap_mode='iou'):
    pred_particles = result_dict['particles']
    
    if overlap_mode == 'chamfer':
        true_particles = [ia for ia in result_dict['truth_particles'] if ia.truth_size > 0]
    else:
        true_particles = [ia for ia in result_dict['truth_particles'] if ia.size > 0]
    
    # Only consider interactions with nonzero predicted nonghost
    matched_particles = []
    
    if matching_mode == 'optimal':
        matched_particles, counts = match_particles_optimal(
            pred_particles, 
            true_particles, 
            min_overlap=min_overlap, 
            overlap_mode=overlap_mode)
        
    if matching_mode == 'one_way':
        if matching_direction == 'pred_to_true':
            matched_particles, counts = match_particles_fn(
                pred_particles, 
                true_particles, 
                min_overlap=min_overlap, 
                overlap_mode=overlap_mode)
        elif matching_direction == 'true_to_pred':
            matched_particles, counts = match_particles_fn(
                true_particles, 
                pred_particles, 
                min_overlap=min_overlap, 
                overlap_mode=overlap_mode)
        
    update_dict = {
        # 'matched_particles': matched_particles,
        'particle_match_values': np.array(counts, dtype=np.float32),
    }

    return update_dict
    


@post_processing(data_capture=['index'], 
                 result_capture=['interactions',
                                 'truth_interactions'])
def match_interactions(data_dict,
                       result_dict,
                       matching_mode='optimal',
                       matching_direction='pred_to_true',
                       match_particles=True,
                       min_overlap=0,
                       overlap_mode='iou'):

    pred_interactions = result_dict['interactions']
    
    if overlap_mode == 'chamfer':
        true_interactions = [ia for ia in result_dict['truth_interactions'] if ia.truth_size > 0]
    else:
        true_interactions = [ia for ia in result_dict['truth_interactions'] if ia.size > 0]
    
    # Only consider interactions with nonzero predicted nonghost
    
    if matching_mode == 'optimal':
        matched_interactions, counts = match_interactions_optimal(
            pred_interactions, 
            true_interactions, 
            min_overlap=min_overlap, 
            overlap_mode=overlap_mode)
        
    if matching_mode == 'one_way':
        if matching_direction == 'pred_to_true':
            matched_interactions, counts = match_interactions_fn(
                pred_interactions, 
                true_interactions, 
                min_overlap=min_overlap, 
                overlap_mode=overlap_mode)
        elif matching_direction == 'true_to_pred':
            matched_interactions, counts = match_interactions_fn(
                true_interactions, 
                pred_interactions, 
                min_overlap=min_overlap, 
                overlap_mode=overlap_mode)
        
    update_dict = {
        # 'matched_interactions': matched_interactions,
        'interaction_match_values': np.array(counts, dtype=np.float32),
    }
    
    return update_dict


# ----------------------------- Helper functions -------------------------------

def match_parts_within_ints(int_matches):
    '''
    Given list of matches Tuple[(Truth)Interaction, (Truth)Interaction], 
    return list of particle matches Tuple[TruthParticle, Particle]. 

    This means rather than matching all predicted particles againts
    all true particles, it has an additional constraint that only
    particles within a matched interaction pair can be considered
    for matching. 
    '''

    matched_particles, match_counts = [], []

    for m in int_matches:
        ia1, ia2 = m[0], m[1]
        num_parts_1, num_parts_2 = -1, -1
        if m[0] is not None:
            num_parts_1 = len(m[0].particles)
        if m[1] is not None:
            num_parts_2 = len(m[1].particles)
        if num_parts_1 <= num_parts_2:
            ia1, ia2 = m[0], m[1]
        else:
            ia1, ia2 = m[1], m[0]
            
        for p in ia2.particles:
            if len(p.match) == 0:
                if type(p) is Particle:
                    matched_particles.append((None, p))
                    match_counts.append(-1)
                else:
                    matched_particles.append((p, None))
                    match_counts.append(-1)
            for match_id in p.match:
                if type(p) is Particle:
                    matched_particles.append((ia1[match_id], p))
                else:
                    matched_particles.append((p, ia1[match_id]))
                match_counts.append(p._match_counts[match_id])
    return matched_particles, np.array(match_counts)


def check_particle_matches(loaded_particles, clear=False):
    match_dict = OrderedDict({})
    for p in loaded_particles:
        for i, m in enumerate(p.match):
            match_dict[int(m)] = p.match_counts[i]
        if clear:
            p._match = []
            p._match_counts = OrderedDict()

    match_counts = np.array(list(match_dict.values()))
    match = np.array(list(match_dict.keys())).astype(int)
    perm = np.argsort(match_counts)[::-1]
    match_counts = match_counts[perm]
    match = match[perm]

    return match, match_counts