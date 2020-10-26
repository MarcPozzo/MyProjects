

*Boucle.py
** est le script tel qui doit tourner sur le raspberry pi

*functions.py 
** est un script de test important dont on se sert pour faire l'extraction des imagettes et les test pour VGG16 ou Lenet.

recouvrement/
** évaluationo du recouvrement entre imagettes identifiées automatiquement et zones annotées selon les paramètres utilisés. Les zones contenant des oiseux sont-elles bien identifiées ? Quelles sont les tailles relatives des imagettes par rapport aux zones annotées ?

test_unitaires/
** précurseur du dossier "recouvrement", regarde la correspondance entre les imagettes identifées et les zones annotées mais de manière plus simple, vérifie aussi que les classes identifiées sont correctes. 

fp_threshold 
** Selon le seuil de probabilité appliqué en sortie du réseau de neurone combien reste-t-il de faux positifs/vrais positifs etc. ?

__pycache__ répertoire généré automatiquement par ???




