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
| `train_data.py` | Entraînement de référence avec ou sans Lipschitz, avec ou sans redimensionnement contrôlé de la résolution de détail (`DIM_RESIZE`, `ORIG_SZ`) : l'image est sous-échantillonnée puis ré-agrandie en interpolation *nearest*, ce qui limite le contenu haute fréquence que le réseau doit apprendre à représenter | Fonctionnel |
| `train_data_dual_head.py` | Correspond à l'architecture **DualHead** du rapport de PFE (Annexe C) : un corps 1-Lipschitz partagé, une tête HKR et une tête EBM non contrainte, combinées par une pondération sigmoïde `α(x) = σ(E_HKR(x))` selon la proximité estimée à la distribution | D'après le rapport, sous-performe nettement en pratique (FID ≈ 80, contre 46 pour la baseline sur CIFAR-10). |
| `train_data_coarsetofine.py` | Entraînement en cascade multi-résolution (grossier → fin), chaque étage initialisant le suivant ; rejoint la piste de Zhao et al. citée section 2.3.3 du rapport comme motivation à l'apprentissage multi-échelle par *downsampling* seulement dans ce code nous ne réduisons que la résolution des détails, pas la résolution de l'image. | à approfondir, les théories et la et les modèles ont les comportements attendus mais aucun résultats probant au niveau du FID. |
| `train_data_dual_models_warmup` | Entrainement avec pré-initialisation Lipschitz avec 2 modèles (section 6.4), 1 modèle apprend d'abord en étant Lipschitz pour rapprocher le buffer de la distribution cible, ensuite on reprend ce buffer et on entraine un réseau non lipschitz dessus (en réinitialisant le buffer plus avec du bruit, mais avec du bruit raffiné par le premier modèle). | Fonctionnel. Très bonnes performances montrées dans le rapport, cependant l'énergie n'est plus proportionnelle à une log-probabilité (on perd les propriétés des EBMs) car il y a 2 modèles, et on rajoute de la complexité |
| `train_data_freqscheduled.py` | Bruit d'initialisation du buffer persistant filtré par bande de fréquence (FFT), avec un curriculum passant d'un bruit purement basse fréquence à un mélange uniforme basse/moyenne/haute fréquence au fil de l'entraînement | Fonctionnel ; le rapport (Annexe E) confirme empiriquement le gain de stabilité apporté par un bruit d'initialisation multi-fréquence plutôt qu'un bruit uniforme haute fréquence, et note qu'une introduction progressive des fréquences donne les meilleurs résultats mais complexifie le pipeline |
| `train_data_maco.py` | Variante par phase (MACO) au lieu de faire le langevin sur l'image directmeent | **Non fonctionnel**, signalé comme tel dans le code|

`nets.py` contient également l'architecture **Semi-Lipschitz** du rapport
(section 4.5) : seules les premières couches (`SpectralConv2d`/`SpectralLinear` +
`GroupSort2`) sont contraintes 1-Lipschitz, on obtient de très bon résultats si on part du bruit filtré FFT.
Il contient aussi un réseau `Halflip` qui possède tout son features extractor Lipschitz, les résultats avec ce modèles sont plutôt mauvais. Il possède aussi des réseaux entièremenet Lipschitz comme `Wide_ResNet_LLip`. 

Chaque variante d'entraînement possède un script d'évaluation associé
(`eval_checkpoint.py`, `eval_checkpoint_coarsetofine.py`,
`eval_checkpoint_dualhead.py`) qui recharge un checkpoint, régénère des
échantillons par MCMC long-terme et recalcule les diagnostics.
---

## 2. Utilisation

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
logging. 

### Évaluation

`eval_checkpoint*.py` retrouve automatiquement le fichier de config sauvegardé
dans `EXP_DIR/code/`, recharge un checkpoint donné, régénère des échantillons par
MCMC long-terme et recalcule le FID.

