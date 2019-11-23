from nltk.corpus import wordnet as wn
from utils.wordnet_similarity import get_scene_synset_dictionary, syn_to_string
from utils.data import classes

import torch

def LSTD(scene_dist, eps, scene_labels, metric,  device="cuda:0", lesk_scores=None, scene_synsets=None, display=False):
    """
    Calculates the sum of (latent space distance / similarity distance) for all scenes w.r.t to current scene_j (exclusive)
    :param scene_dist: the model's tracked scene distributions
    :param eps:  the model's seeded epsilon
    :param scene_synsets: dictionary of scene synsets
    :param scene_labels: labels from the current batch
    :param metric: Similarity metric to use (Default is Lesk)
        To change: set metric="wup" to use wup similary, etc
    :return: Summed latent space/ similarity distance
    """
    # All LSTD metrics for the current batch
    lstd = torch.tensor([])

    # Find the scene synset if not provided
    if scene_synsets is None:
        scene_synsets = get_scene_synset_dictionary(classes)

    for scene_j in scene_labels:
        scene_j = scene_j.item()

        if display:
            print(f'\n\n{classes[scene_j]}\n{"=" * 48}')

        # LSTD metric for all adjacent scenes w.r.t scene_j
        scene_lstd = torch.tensor([])
        scene_lstd = scene_lstd.to(device)

        # for scene_k in scene_dist.keys():
        for scene_k in range(len(classes)):
            if scene_j != scene_k:
                # Scene Latent Space Distance
                uj = scene_dist[scene_j][0] + scene_dist[scene_j][1] * eps
                uk = scene_dist[scene_k][0] + scene_dist[scene_k][1] * eps
                latent_dist = torch.abs(uj - uk)

                # Calculate Tree Distance Between Scenes
                syn_j = scene_synsets[classes[scene_j]]
                syn_k = scene_synsets[classes[scene_k]]

                if metric == "wup":
                    tree_dist = 1 - wn.wup_similarity(syn_j, syn_k)
                else:
                    # tree_dist = 1 - lesk_simarity(syn_j, syn_k)
                    tree_dist = lesk_scores[syn_to_string(syn_j)].values[scene_k]


                # Distance for this specific iteration
                cur_lstd = latent_dist / tree_dist
                if tree_dist == 0:
                    cur_lstd = torch.zeros(1, device=device)
            elif classes[scene_j] == classes[scene_k]:
                cur_lstd = torch.zeros(1, device=device)
            # Add this distance to find cummalitve distance w.r.t scene_j
            scene_lstd = torch.cat((scene_lstd, cur_lstd))


            if display:
                print(f'Calculated distance: {classes[scene_j]} --> {classes[scene_k]} = {cur_lstd.sum()}')

        # Calculate the Sum of the distances of all scene w.r.t to scenej
        scene_lstd = torch.tensor([torch.sum(scene_lstd)])

        if display:
            print(f'Summed scene_lstd: {scene_lstd.item()}')

        # Reshape empty tensor accordingly
        if lstd.size()[0] == 0:
            lstd = lstd.view(*lstd.shape)
        lstd = torch.cat((lstd, scene_lstd))

    # if metric == "wup":
    return torch.log(torch.sum(lstd))
    # return torch.sum(lstd)