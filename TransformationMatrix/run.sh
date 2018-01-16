#!/bin/bash

array=(20 50 100 150 200 250 500 800 1000 1500 2000 )
for data in ${array[@]}
do
    python3 LinearMapping.py ${data}
    python3 /disk/xfbai/mywork/Bilingual-lexicon-survey/wsim/eval-word-vectors/all_wordsim.py  LinearMappingres.en /disk/xfbai/mywork/Bilingual-lexicon-survey/wsim/eval-word-vectors/data/word-sim/
    python3 /disk/xfbai/mywork/Bilingual-lexicon-survey/vecmap/eval_translation.py -d /disk/xfbai/mywork/Bilingual-lexicon-survey/vecmap/data/dictionaries/en.de.dict --type 2 LinearMappingres.en LinearMappingres.de
done
