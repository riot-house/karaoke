# Studio Karaoké — le site

Site **entièrement statique** : trois fichiers, aucun serveur applicatif,
aucune base de données. Tout tourne dans le navigateur du visiteur.

| Fichier | Rôle |
|---|---|
| `index.html` | toute l'application : interface, moteur de calage, encodeur vidéo |
| `chant-asm.mp3` | le morceau chargé par défaut à l'ouverture |
| `karaoke-asr.js` | la reconnaissance des paroles (Whisper), chargée au premier clic |
| `404.html` | page d'erreur |

## Différences avec la page publiée sur claude.ai

- **Mémoire** : le stockage du navigateur (`localStorage`) au lieu du service
  de claude.ai. Elle reste sur la machine du visiteur ; chaque navigateur a
  la sienne.
- **Téléchargements** : un lien de téléchargement classique, au lieu du
  mécanisme d'artefact. La vidéo et le JSON s'enregistrent normalement.
- **Audio** : le MP3 est un fichier à part, mis en cache par le navigateur,
  au lieu d'être encapsulé dans le HTML. La page passe de 2,1 Mo à 131 Ko.
- **Reconnaissance des paroles** : le bouton n'existe que sur cette version,
  puisque le modèle ne peut pas être téléchargé depuis un artefact.

## Changer le morceau par défaut

1. Remplacez `chant-asm.mp3` (gardez le même nom, ou changez `SONG_URL`
   dans `index.html`).
2. Dans `index.html`, remplacez `DEMO_LYRICS` par les nouvelles paroles et
   `SONG_TIMINGS` par le JSON exporté depuis le studio (bouton
   « Exporter le JSON du plugin »).

Sans ces deux mises à jour, la page affichera les anciennes paroles sur le
nouveau son.

## Reconnaissance des paroles — les deux leviers

Le bouton **Reconnaître les paroles** a deux réglages, comme le panneau
After Effects.

**Le modèle**, c'est-à-dire la finesse d'écoute. Tailles réelles des
fichiers téléchargés, une seule fois puis gardés en cache :

| Choix | Modèle | Téléchargement | Vitesse | Sur du chant |
|---|---|---|---|---|
| Rapide | whisper-tiny_timestamped | 41 Mo | référence | souvent insuffisant |
| Équilibré | whisper-base_timestamped | 77 Mo | ~1,5× plus lent | correct sur voix dégagée |
| Précis | whisper-small_timestamped | 249 Mo | ~3× plus lent | **le vrai saut de qualité** |

`medium` et `large-v3` (800 Mo à 1,6 Go) ne sont pas proposés : ils ne
tiennent pas dans un navigateur.

**Isoler le centre.** Dans un enregistrement, la voix est presque
toujours au centre — identique à gauche et à droite — alors que
l'ambiance et les instruments larges diffèrent entre les canaux. En
comparant les deux canaux fréquence par fréquence, on atténue ce qui est
large. Mesuré sur le chant ASM : 11,5 dB de contenu retiré, corrélation
0,983 avec le mono (c'est bien la même voix), 1,2 s de calcul pour 67 s
d'audio. Sans effet sur un enregistrement mono.

**Le pourcentage affiché est le seul juge utile** : c'est la part de vos
paroles réellement retrouvées dans l'audio. En dessous de 55 %, montez
d'un cran. Le repli « Ce que la machine a entendu » montre la
transcription brute — si elle est vide ou absurde, aucun réglage de
calage n'y changera rien, et le calage au clavier reste la bonne réponse.

## Deux pièges à ne pas réintroduire

Si vous touchez un jour à `karaoke-asr.js`, ces deux points sont des
sources de panne silencieuse :

- **Les modèles DOIVENT porter le suffixe `_timestamped`.** Seules ces
  variantes sont exportées avec les attentions croisées du décodeur.
  Sans elles, la transcription marche mais la datation des mots échoue
  avec « Model outputs must contain cross attentions to extract
  timestamps ».
- **`chunk_length_s` doit valoir 29, pas 30.** À exactement 30, la
  datation dégénère : tous les mots d'une tranche reçoivent le même
  instant, et le calage devient aberrant sans qu'aucune erreur ne soit
  levée. C'est un bug connu de transformers.js.

## À vérifier une fois en ligne

- Le site doit être servi en **HTTPS** : les modules ES et WebGPU l'exigent,
  et l'encodage vidéo aussi.
- Le premier clic sur « Reconnaître les paroles » télécharge le modèle
  (~75 Mo) : c'est normal, et il est ensuite gardé en cache.
