import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer
from pypinyin import pinyin, Style, load_phrases_dict 

def _pre_pinyin_setting():
    ''' fix pinyin error'''
    load_phrases_dict({'嗯':[['ēn']]})
    load_phrases_dict({'不破不立':[['bù'], ['pò'], ['bù'], ['lì']]})

def generate_pinyin(txtfile):
    with open(txtfile) as fid:
        txtlines = [x.strip() for x in fid.readlines()]
    split_sentences = re.split(r'[，。,.]', txtlines[0])
    pinyin_sentences = []
    for sentence in split_sentences:
        pinyin_list = pinyin(sentence, style=Style.TONE3)
        pinyin_list = [x[0]+'5' if not x[0][-1].isdigit() else x[0] for x in
            pinyin_list]
        pinyin_sentences.append(' '.join(pinyin_list))
    return pinyin_sentences

def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)

def run_eval(args):
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
    sentences = generate_pinyin(args.txtfile)
    for i, text in enumerate(sentences):
        path = '%s-%d.wav' % (base_path, i)
        print('Synthesizing: %s' % path)
        with open(path, 'wb') as f:
            f.write(synth.synthesize(text))
    wav_name_list = ['%s-%d.wav'%(base_path, i) for i in range(len(sentences))]
    wav_name = ' '.join(wav_name_list)
    os.system('sox %s synthesis.wav' % (wav_name))
  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('txtfile', help='Path to mandarin txt file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='', 
            help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
    main()
