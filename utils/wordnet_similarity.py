from nltk.corpus import wordnet as wn

import numpy as np
import pandas as pd
import re
import subprocess

def get_scene_synset_dictionary(classes):
    """
    :param classes: list of string value class names
    :return:  a dictionary of scenes and their corresponding synsets
    """
    scene_synsets = {}

    for name in classes:
        w = wn.synsets(name)
        if not wn.synsets(name):
            synsets = name.split('_')
            w = [wn.synsets(i)[0] for i in synsets]
        scene_synsets[name] = w[0]
    return scene_synsets


def convert_synsets(classes):
    """
        Converts Python nltk synsets to be read by perl metric scripts.

        :param classes: list of class labels for each scene category.

        Example: Synset('water_tower.n.01') becomes water_tower#n#01.

        Returns: List of parsed strings to be used when calling the perl similarity script
    """

    d = get_scene_synset_dictionary(classes)
    parse_syns = []

    for syn in d.values():
        syn = str(syn).replace("Synset('", "")
        syn = syn.replace(".", "#")
        syn = syn.replace("0", "")
        syn = syn.replace("')", "")
        parse_syns.append(syn)

    return parse_syns


def syn_to_string(syn):
    """
    :param syn: synset object to convert to string
    :return: String value of synset
    """
    syn = str(syn).replace("Synset('", "")
    syn = syn.replace(".", "#")
    syn = syn.replace("0", "")
    syn = syn.replace("')", "")
    return syn


def lesk_simarity(synj, synk):
    """
    Use to calculate the lesk similarity between class labels provided.

    :param synj, synk: Synsets of corresponding class labels to read by perl script. NOTE: Will work with nltk synset objects
    Returns: Lesk Similarity Metric
    """
    # Prep the synsets to be parsed by the Perl script
    synj = str(synj).replace("Synset('", "")
    synk = str(synk).replace("Synset('", "")
    synj = synj.replace(".", "#")
    synk = synk.replace(".", "#")
    synj = synj.replace("0", "")
    synk = synk.replace("0", "")
    synj = synj.replace("')", "")
    synk = synk.replace("')", "")

    # Call the Similarity script with the two classes of interest
    cmd = f"similarity.pl --type 'WordNet::Similarity::lesk' {synj} {synk}"
    output = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)

    # Parse the output and store the value as a float
    output = output.stdout.read()
    parsed = re.split('\s+', str(output))
    metric_score = int(parsed[-1][:-3]) / 1.0
    return metric_score

def exhuastive_lesk_simarity_metric(classes, filename="lesk_scores.csv"):
    """
    Use to calculate the lesk similarity(not distance) w.r.t to each class label provided.

    :param classes: List of class labels for each scene category
    Returns: A dictionary of lesk similarity scores stored in a numpy arrays (where keys are the class label and values are the corresponding numpy array)
    """
    # Prep the synsets to be parsed by the Perl script
    parse_syns = convert_synsets(classes)
    lesk_metrics = {}

    for synj in parse_syns:
        cross_metric = np.array([])
        for synk in parse_syns:

            if synj != synk:

                # Call the Similarity script with the two classes of interest
                cmd = f"similarity.pl --type 'WordNet::Similarity::lesk' {synj} {synk}"
                output = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)

                # Parse the output and store the value as a float
                output = output.stdout.read()
                parsed = re.split('\s+', str(output))
                metric_score = int(parsed[-1][:-3]) / 1.0
                print(f'Lesk distance {synj} ---> {synk}: {metric_score}')
            else:  # Same class set current score to 0 (synj == synk)
                metric_score = 0

            # Append the current score to the current cross metric array (w.r.t synj's class)
            cross_metric = np.append(cross_metric, metric_score)

        # Save and store results in dictionary
        lesk_metrics[synj] = cross_metric

    # Export Dictionary to csv file for future use
    # Return Dictionary
    lesk = pd.DataFrame(lesk_metrics)
    lesk.to_csv(filename)
    return lesk_metrics