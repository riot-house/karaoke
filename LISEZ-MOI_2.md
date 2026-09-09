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

## À vérifier une fois en ligne

- Le site doit être servi en **HTTPS** : les modules ES et WebGPU l'exigent,
  et l'encodage vidéo aussi.
- Le premier clic sur « Reconnaître les paroles » télécharge le modèle
  (~75 Mo) : c'est normal, et il est ensuite gardé en cache.
