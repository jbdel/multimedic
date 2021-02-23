import os
from .cocoeval import Rouge
import shutil
import pathlib
import subprocess


def meteor_score(refs, hyps, language="en"):
    jar = "./torchnmt/scorers/cocoeval/meteor/meteor-1.5.jar"
    __cmdline = ["java", "-Xmx2G", "-jar", jar,
                 "-", "-", "-stdio"]
    assert os.path.exists(jar), 'meteor jar not found'
    env = os.environ
    env['LC_ALL'] = 'en_US.UTF-8'

    # Sanity check
    if shutil.which('java') is None:
        raise RuntimeError('METEOR requires java which is not installed.')

    cmdline = __cmdline[:]
    refs = [refs] if not isinstance(refs, list) else refs

    if isinstance(hyps, str):
        # If file, open it for line reading
        hyps = open(hyps)

    if language == "auto":
        # Take the extension of the 1st reference file, e.g. ".de"
        language = pathlib.Path(refs[0]).suffix[1:]

    cmdline.extend(["-l", language])

    # Make reference files a list
    iters = [open(f) for f in refs]
    iters.append(hyps)

    # Run METEOR process
    proc = subprocess.Popen(cmdline,
                            stdout=subprocess.PIPE,
                            stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=env,
                            universal_newlines=True, bufsize=1)

    eval_line = 'EVAL'

    for line_ctr, lines in enumerate(zip(*iters)):
        lines = [l.rstrip('\n') for l in lines]
        refstr = " ||| ".join(lines[:-1])
        line = "SCORE ||| " + refstr + " ||| " + lines[-1]

        proc.stdin.write(line + '\n')
        eval_line += ' ||| {}'.format(proc.stdout.readline().strip())

    # Send EVAL line to METEOR
    proc.stdin.write(eval_line + '\n')

    # Dummy read segment scores
    for i in range(line_ctr + 1):
        proc.stdout.readline().strip()

    # Compute final METEOR
    try:
        score = float(proc.stdout.readline().strip())
    except Exception as e:
        score = 0.0
    finally:
        # Close METEOR process
        proc.stdin.close()
        proc.terminate()
        proc.kill()
        proc.wait(timeout=2)
        return score


def rouge_score(refs, hyps):
    assert len(hyps) == len(refs), "ROUGE: # of sentences does not match."

    rouge_scorer = Rouge()

    rouge_sum = 0
    for hyp, ref in zip(hyps, refs):
        rouge_sum += rouge_scorer.calc_score([hyp], [ref])

    score = (rouge_sum) / len(hyps)

    return score


def bleu_score(refs, hyps):
    # -*- coding: utf-8 -*-
    import subprocess
    import pkg_resources
    BLEU_SCRIPT = './torchnmt/scorers/multi-bleu.perl'
    cmdline = [BLEU_SCRIPT]

    # Make reference files a list
    refs = [refs] if not isinstance(refs, list) else refs
    cmdline.extend(refs)

    if isinstance(hyps, str):
        hypstring = open(hyps).read().strip()
    elif isinstance(hyps, list):
        hypstring = "\n".join(hyps)

    score = subprocess.run(cmdline, stdout=subprocess.PIPE,
                           input=hypstring,
                           universal_newlines=True).stdout.splitlines()

    if len(score) == 0:
        return 0.0
    else:
        score = score[0].strip()
        float_score = float(score.split()[2][:-1])
        verbose_score = score.replace('BLEU = ', '')
        return float_score


def compute_scores(refs, hyps, base):
    assert len(refs) == len(hyps)

    # Dump
    with open(base.format('refs.txt'), 'w') as f:
        f.write('\n'.join(refs))

    with open(base.format('hyps.txt'), 'w') as f:
        f.write('\n'.join(hyps))

    scores = {
        "BLEU": bleu_score(base.format('refs.txt'), base.format('hyps.txt')),
        "ROUGE": rouge_score(refs, hyps) * 100,
        "METEOR": meteor_score(base.format('refs.txt'), base.format('hyps.txt')) * 100,
    }
    scores = {k: round(v, 2) for k, v in scores.items()}

    return scores
