torch 1.4.0  !!!

train

```
for i in {1..5}; 
do 
    python bin/train.py config/mediaca/rnn.yml
done
```

todo ensembling of all models:
```
python bin/ensemblor.py config/mediaca/rnn.yml
```
it is possible to override options:

```
python bin/ensemblor.py config/mediaca/rnn.yml ensemblor.beam_width:16
```

How does ensemblor.mode works:<br/>
ensemblor.mode: all : takes all ckpt for ensembling<br/>
ensemblor.mode: best-n : takes n best checkpoints for ensembling<br/>

example:
```
python bin/ensemblor.py config/mediaca/rnn.yml ensemblor.beam_width:16 ensemblor.mode:best-2
```
takes the two best checkpoints for ensembling
