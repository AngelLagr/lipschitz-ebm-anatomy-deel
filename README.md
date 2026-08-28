# Lipschitz EBM Anatomy

Étude de la stabilisation de l'apprentissage ML d'Energy-Based Models (EBM) par
contrainte de Lipschitz sur le réseau d'énergie.

Base du code : [ebm-anatomy](https://github.com/point0bar1/ebm-anatomy) (Nijkamp, Hill,
Han, Zhu, Wu — *On the Anatomy of MCMC-Based Maximum Likelihood Learning of
Energy-Based Models*, AAAI 2020, [arXiv:1903.12370](https://arxiv.org/abs/1903.12370)).
Les contraintes de Lipschitz sont implémentées via la librairie
[deel-torchlip](https://github.com/deel-ai/deel-torchlip) et, nous avons testé les convolutions
exactement orthogonales, via [orthogonium](https://github.com/deel-ai/orthogonium).

Ce dépôt constitue le support de code du stage de fin d'études *Stabilisation des
Modèles Basés sur l'Énergie grâce à la Contrainte de Lipschitz* (Angel Lagrange,
PFE ANITI / équipe DEEL, IRT Saint Exupéry, encadrants Victor Boutin et Franck
Mamalet, 2026), dont le rapport complet (ci-après « le rapport de PFE ») sera
versé dans ce dépôt. Ce README n'en donne qu'un résumé orienté code.

---

## 1. Codes

| Fichier | Idée testée | Statut |
|---|---|---|
| `train_data.py` | Entraînement de référence, avec redimensionnement contrôlé de la résolution de détail (`DIM_RESIZE`, `ORIG_SZ`) : l'image est sous-échantillonnée puis ré-agrandie en interpolation *nearest*, ce qui limite le contenu haute fréquence que le réseau doit apprendre à représenter | Fonctionnel |
| `train_data_dual_head.py` | Correspond à l'architecture **DualHead** du rapport de PFE (Annexe C) : un corps 1-Lipschitz partagé, une tête HKR et une tête EBM non contrainte, combinées par une pondération sigmoïde `α(x) = σ(E_HKR(x))` selon la proximité estimée à la distribution | D'après le rapport, sous-performe nettement en pratique (FID ≈ 80, contre 46 pour la baseline sur CIFAR-10) sans rééquilibrage explicite des gradients entre les deux têtes (éq. 29 du rapport) et régularisation de la tête EBM — non systématiquement actifs dans ce script |
| `train_data_coarsetofine.py` | Entraînement en cascade multi-résolution (grossier → fin), chaque étage initialisant le suivant ; rejoint la piste de Zhao et al. citée section 2.3.3 du rapport comme motivation à l'apprentissage multi-échelle par *downsampling* | Fonctionnel |
| `train_data_freqscheduled.py` | Bruit d'initialisation du buffer persistant filtré par bande de fréquence (FFT), avec un curriculum passant d'un bruit purement basse fréquence à un mélange uniforme basse/moyenne/haute fréquence au fil de l'entraînement | Fonctionnel ; le rapport (Annexe E) confirme empiriquement le gain de stabilité apporté par un bruit d'initialisation multi-fréquence plutôt qu'un bruit uniforme haute fréquence, et note qu'une introduction progressive des fréquences donne les meilleurs résultats mais complexifie le pipeline |
| `train_data_maco.py` | Variante par phase (MACO) | **Non fonctionnel**, signalé comme tel dans le code (« isn't working for now and requires more investigation ») ; non mentionné dans le rapport de PFE |

`nets.py` contient également l'architecture **Semi-Lipschitz** du rapport
(section 4.5) : seules les premières couches (`SpectralConv2d`/`SpectralLinear` +
`GroupSort2`) sont contraintes 1-Lipschitz, le reste du réseau restant libre — cf.
`VanillaNetHalfLip` et le paramètre `halflip` de `Wide_ResNet_LLip`. C'est la
méthode que le rapport recommande comme meilleur compromis simplicité/coût,
au prix d'un léger écart de FID par rapport à la pré-initialisation séquentielle
(section 4.4 du rapport, deux réseaux entraînés successivement) — cette dernière
n'est pas incluse comme script unique dans ce dépôt : elle s'obtient en
enchaînant deux exécutions de `train_data.py` (un modèle Lipschitz, puis un
modèle non contraint dont le buffer est amorcé par le premier).

Chaque variante d'entraînement possède un script d'évaluation associé
(`eval_checkpoint.py`, `eval_checkpoint_coarsetofine.py`,
`eval_checkpoint_dualhead.py`) qui recharge un checkpoint, régénère des
échantillons par MCMC long-terme et recalcule les diagnostics.

---

## 4. Diagnostics additionnels au projet

Au-delà de `d_s_t` et `r_s_t` du papier original, ce dépôt ajoute :

- **FID (Fréchet Inception Distance)** (`utilfid.py`) entre échantillons positifs
  et échantillons générés, calculé périodiquement (`log_freq`) : extracteur CNN
  dédié pour MNIST (poids pré-entraînés locaux dans fidcheckpoints), features InceptionV3 pour
  CIFAR-10.
- **Indice d'instabilité** (`compute_instability_index`) : coefficient de
  variation (écart-type / moyenne absolue) de `f(x_q)` et `f(x_s_t)` sur une
  fenêtre glissante courte (2 itérations), cette métrique n'est pas particulièrement
  pertinente ni fondée théoriquement.

Tous ces indicateurs sont tracés par `plot_diagnostics` et sauvegardés en CSV
(`training_metrics.csv`) et dans les checkpoints complets
(`checkpoints/last_checkpoint.pth`), qui permettent une reprise exacte de
l'entraînement (poids, optimiseur, buffer persistant `s_t_0`, tous les
historiques de métriques).

---

## 5. Utilisation

### Entraînement

Dans le script d'entraînement choisi (`train_data*.py`), définir en tête de
fichier :

```python
EXP_DIR = './out_data/Expe/nom_de_l_experience/'
CONFIG_FILE = './config_locker/nom_config.json'
```

puis exécuter le script. Un dossier `EXP_DIR` est créé avec les sous-dossiers
`checkpoints/`, `shortrun/`, `longrun/`, `plots/`, `code/` (copie du code et de
la config utilisés, pour la reproductibilité).

### Fichiers de configuration (`config_locker/`)

Chaque fichier JSON définit intégralement une expérience : jeu de données
(`mnist`, `cifar10`, `flowers`), architecture (`net_type`, clé de
`net_bank` dans `train_data.py`), `Lip_const` initiale, optimiseur, paramètres de
Langevin (`epsilon`, `num_shortrun_steps`, `shortrun_init`), et fréquences de
logging. Les fichiers nommés `mnist_Lip{K}_Langevin{L}_Seed42*.json` forment un
balayage systématique de la constante de Lipschitz `K` et du nombre de pas de
Langevin `L`, correspondant à l'étude paramétrique menée dans ce projet ; les
suffixes `Monitor`, `Constrained`, `Ortho`, `ADAM` indiquent respectivement
l'activation du monitoring de stabilité, de la mise à jour adaptative de
`Lip_const`, de l'usage des convolutions orthogonales, et de l'optimiseur Adam à
la place de SGD.

### Évaluation

`eval_checkpoint*.py` retrouve automatiquement le fichier de config sauvegardé
dans `EXP_DIR/code/`, recharge un checkpoint donné, régénère des échantillons par
MCMC long-terme et recalcule le FID.

