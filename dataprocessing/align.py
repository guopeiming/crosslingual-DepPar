import os
import random
from typing import List, Tuple, Dict, Union, Iterator
import matplotlib.pyplot as plt
import numpy as np
import json


def filter_raw_corpus(sentences_file: str, tokens_file: str, output_dir: str, src: str, tgt: str):
    sents_set = set()
    with open(sentences_file, 'r', encoding='utf-8') as s_align_reader,\
        open(tokens_file, 'r', encoding='utf-8') as t_align_reader,\
        open(os.path.join(output_dir, f'{src}.txt'), 'w', encoding='utf-8') as src_writer,\
        open(os.path.join(output_dir, f'{tgt}.txt'), 'w', encoding='utf-8') as tgt_writer,\
        open(os.path.join(output_dir, f'{src}-{tgt}.aligns'), 'w', encoding='utf-8') as aligns_writer:

        for i, (sentences, token_aligns) in enumerate(zip(s_align_reader, t_align_reader)):

            if i % 5 != 0:
                continue

            sent_src, sent_tgt = sentences.split('|||')
            sent_src, sent_tgt = sent_src.strip().split(' '), sent_tgt.strip().split(' ')
            token_aligns = token_aligns.strip().split(' ')

            if ' '.join(sent_src) in sents_set:
                continue

            if 8 < len(sent_src) < 128 and 8 < len(sent_tgt) < 128 and len(token_aligns) > 4:
                src_writer.write(' '.join(sent_src)+'\n')
                tgt_writer.write(' '.join(sent_tgt)+'\n')
                aligns_writer.write(' '.join(token_aligns)+'\n')
                sents_set.add(' '.join(sent_src))
            if len(sents_set) >= 80000:
                break

    print(f'finish.\ntotal {len(sents_set)} sentences.')


def tagged_file_generator(file: str) -> Iterator[Tuple[List[str], List[str]]]:
    with open(file, 'r', encoding='utf-8') as reader:
        tokens, postags = [], []
        for line in reader:
            line = line.strip()
            if line.startswith('# '):
                continue
            if line == '':
                yield tokens, postags
                tokens, postags = [], []
            else:
                line = line.split('\t')
                tokens.append(line[1])
                postags.append(line[3])


def compute_alignment_difference(
    align_file: str, return_aligns: bool = False
) -> Dict[str, List[Union[int, str]]]:
    diffs, aligns_res = [], []
    with open(align_file, 'r', encoding='utf-8') as align_reader:
        for line in align_reader:
            if return_aligns:
                aligns_res.append(line.strip())

            aligns = line.strip().split(' ')
            diff = 0
            for align in aligns:
                i, j = int(align.split('-')[0]), int(align.split('-')[1])
                diff += abs(i-j)
            diffs.append(diff/len(aligns))

    res = {'diffs': diffs}
    if return_aligns:
        res['aligns'] = aligns_res
    return res


def split_data(lang: str):
    writer = None
    with open(f"./data/data2/origin/{lang}/{lang}.txt", "r", encoding="utf-8") as reader:
        for i_snt, line in enumerate(reader):
            if i_snt % 20000 == 0:
                if writer is not None:
                    writer.close()
                writer = open(f"./data/{lang}{i_snt//20000}.txt", "w", encoding="utf-8")
                print(f"sentence from {i_snt} to {i_snt+20000} will be written to ./data/{lang}{i_snt//20000}.txt")
            writer.write(line)
    print("sentences finished.")


def merge_data(lang: str):
    with open(f"./data/data2/origin/{lang}/{lang}.tag.txt", "w", encoding="utf-8") as writer:
        for i in range(4):
            with open(f"./data/{lang}{i}.tag.txt", "r", encoding="utf-8") as reader:
                for line in reader:
                    writer.write(line)


def check_tagged_file(raw_file: str, tagged_file: str) -> None:
    num = 0
    tgt_tag_reader = tagged_file_generator(tagged_file)

    with open(raw_file, "r", encoding='utf-8') as tgt_reader:
        for snt_raw, (tokens, postags) in zip(tgt_reader, tgt_tag_reader):
            snt_raw = snt_raw.strip()
            assert snt_raw == ' '.join(tokens), snt_raw
            num += 1

    print(f'total {num} sentences, checking finished.')


def convert_conll2jsonl(tagfile: str, alignfile: str, src: str, tgt: str):
    tag_snts = list(tagged_file_generator(tagfile))
    align_diffs_res = compute_alignment_difference(alignfile, True)
    indexs = np.argsort(np.array(align_diffs_res['diffs']))
    aligns = align_diffs_res['aligns']

    file_dir = os.path.dirname(tagfile)
    with open(os.path.join(file_dir, f'{tgt}.jsonl'), 'w', encoding='utf-8') as unsorted_writer,\
        open(os.path.join(file_dir, f'{tgt}.sorted.jsonl'), 'w', encoding='utf-8') as sorted_writer,\
        open(os.path.join(file_dir, f'{src}-{tgt}.sorted.aligns'), 'w', encoding='utf-8') as aligns_writer:

        for unsorted_i, sorted_i in enumerate(indexs):
            tokens, postags = tag_snts[unsorted_i]
            unsorted_writer.write(json.dumps({"tokens": tokens, "postags": postags}, ensure_ascii=False)+"\n")

            tokens, postags = tag_snts[sorted_i]
            sorted_writer.write(json.dumps({"tokens": tokens, "postags": postags}, ensure_ascii=False)+"\n")

            aligns_writer.write(aligns[sorted_i]+'\n')

    print(f'unsorted data is saved in {tgt}.jsonl.')
    print(f'sorted data is saved in {tgt}.sorted.jsonl, {src}-{tgt}.sorted.aligns')


def compute_statistics(input_file: str):
    diffs = compute_alignment_difference(input_file)['diffs']
    diffs = np.array(diffs)

    plt.hist(diffs, bins=25)
    plt.xlabel('align_diff')
    plt.ylabel('sameple num')
    plt.show()

    print(diffs[3999], diffs[7999], diffs[11999], diffs[15999], diffs[19999])
    print(
        np.sum(diffs <= 1.0),
        np.sum((1.0 < diffs) & (diffs <= 2.0)),
        np.sum((2.0 < diffs) & (diffs <= 3.0)),
        np.sum((3.0 < diffs) & (diffs <= 4.0)),
        np.sum(diffs > 4.0)
    )


if __name__ == '__main__':
    # filter_raw_corpus(
    #     './data/europarl/en-de.align', './data/europarl/en-de.token.align',
    #     './data/data2/origin/de/', 'en', 'de'
    # )

    # split_data("de")

    # merge_data("de")

    # check_tagged_file(
    #     './data/data2/origin/de/de.txt', './data/data2/origin/de/de.tag.txt'
    # )

    # convert_conll2jsonl(
    #     "./data/data2/origin/de/de.tag.txt", "./data/data2/origin/de/en-de.aligns", "en", "de"
    # )

    # compute_statistics('./data/de/origin/en-de.sorted.aligns')
