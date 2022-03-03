from overrides import overrides
from typing import Dict, Iterator, List, Tuple
import json
from functools import reduce
from operator import mul
import os


def compute_alignment_differences(align_str: str):
    aligns = align_str.split(" ")
    align_diff = 0.

    for align in aligns:
        i, j = align.split("-")
        align_diff += abs(int(i) - int(j))

    align_diff = align_diff/len(aligns)
    return align_diff


class Prediction():

    def __init__(
        self,
        rawdata_file: str, labeleddata_file: str, leftdata_file: str,
        align_file: str, leftalign_file: str,
        conf_threshold: float, aligndiff_threshold: float,
        test_lang: str, train_lang: str,
    ) -> None:
        super().__init__()
        self.rawdata_file = rawdata_file
        self.labeleddata_file = labeleddata_file
        self.leftdata_file = leftdata_file

        self.align_file = align_file
        self.leftalign_file = leftalign_file

        self.test_lang = test_lang
        self.train_lang = train_lang

        self.conf_threshold = conf_threshold
        self.aligndiff_threshold = aligndiff_threshold

    def filtered_snts(self, snts: List[Dict]):
        filtered_snts = []
        aligns = self.get_aligns()
        if len(aligns) != len(snts):
            raise ValueError(
                f"the num of alignment differences:{len(aligns)}\
                     and sentences:{len(snts)} are not equal."
            )

        data_writer = open(self.leftdata_file, "w", encoding="utf-8")
        align_writer = open(self.leftalign_file, "w", encoding="utf-8")

        for snt, align in zip(snts, aligns):
            confidence_score = reduce(mul, snt["confidences"])
            align_diff = compute_alignment_differences(align)
            if (confidence_score > self.conf_threshold) and (align_diff <= self.aligndiff_threshold):
                filtered_snts.append(snt)
            else:
                data_writer.write(json.dumps({
                    "tokens": snt["tokens"],
                    "postags": snt["postags"]
                }, ensure_ascii=False)+"\n")
                align_writer.write(align+"\n")

        data_writer.close()
        align_writer.close()

        print(f"the num of the filtered sentences is {len(filtered_snts)}")
        return filtered_snts

    def get_aligns(self) -> List[str]:
        aligns = []
        with open(self.align_file, "r", encoding="utf-8") as reader:
            for line in reader:
                aligns.append(line.strip())
        return aligns

    def writing_snts(self, snts: List[Dict]) -> None:
        with open(self.labeleddata_file, 'a', encoding='utf-8') as writer:
            print(f'append sentences to {self.labeleddata_file}')
            print(f"please check that language will be overrided to {self.train_lang}.")

            for snt in snts:
                writer.write(json.dumps({
                    "tokens": snt['tokens'],
                    "postags": snt['postags'],
                    "heads": snt['heads'],
                    "deprels": snt['deprels'],
                    "confidences": snt['confidences'],
                    "language": self.train_lang,
                }, ensure_ascii=False)+'\n')
            print(f'{len(snts)} sentences were written to {self.labeleddata_file}')

    def jsonl_reader(
        self,
        inputfile: str,
        override_lang: str = None,
    ) -> Iterator[Dict]:
        print(f"reading data from {inputfile}")
        if override_lang is not None:
            print(f'please check that language will be overrided to {override_lang}')

        with open(inputfile, 'r', encoding='utf-8') as reader:
            for line in reader:
                data = json.loads(line.strip())

                if override_lang:
                    data['language'] = override_lang

                yield data

    def rawdata_processing(self):
        raise NotImplementedError()

    def processing(self):
        raise NotImplementedError()


class PipelinePrediction(Prediction):

    def __init__(
        self,
        model_inputfile: str, model_outputfile: str,
        rawdata_file: str, labeleddata_file: str, leftdata_file: str,
        align_file: str, leftalign_file: str,
        conf_threshold: float, aligndiff_threshold: float,
        test_lang: str, train_lang: str,
    ) -> None:
        super().__init__(
            rawdata_file, labeleddata_file, leftdata_file,
            align_file, leftalign_file,
            conf_threshold, aligndiff_threshold,
            test_lang, train_lang
        )
        self.model_inputfile = model_inputfile
        self.model_outputfile = model_outputfile

    @overrides
    def rawdata_processing(self):
        num = 0
        with open(self.model_inputfile, 'w', encoding='utf-8') as writer:
            for snt in self.jsonl_reader(self.rawdata_file, override_lang=self.test_lang):
                writer.write(json.dumps(snt, ensure_ascii=False)+'\n')
                num += 1
        print(f"{num} sentences were writted to {self.model_inputfile}")

    @overrides
    def processing(self):
        snts_p = list(self.jsonl_reader(self.model_outputfile))

        snts_p = self.filtered_snts(snts_p)
        self.writing_snts(snts_p)
        print('finish')


def jsonl_reader(inputfile: str, override_lang: str = None) -> List[Dict]:
    if override_lang is not None:
        print(f'please check that language will be overrided to {override_lang}')

    snts = []
    with open(inputfile, 'r', encoding='utf-8') as reader:
        for line in reader:
            snt = json.loads(line.strip())
            if override_lang is not None:
                snt["language"] = override_lang
            snts.append(snt)

    print(f"reading {len(snts)} sentences from {inputfile}")
    return snts


def prepare_predict_input(
    rawcorpus: str,
    outputfile: str,
    lang: str,
    snt_start: int = None,
    snt_end: int = None
) -> None:
    snts = jsonl_reader(rawcorpus, override_lang=lang)
    if snt_start is not None:
        snts = snts[snt_start: snt_end]
        print(f"filtering sentences from {snt_start} to {snt_end}")
    writing_jsonl(snts, "w", outputfile)


def filtering(
    snts: List[Dict],
    snts_num: int,
) -> Tuple[List[Dict], List[Dict]]:
    snts = sorted(snts, key=lambda inst: reduce(mul, inst['confidences']), reverse=True)
    return snts[:snts_num], snts[snts_num:]


def writing_jsonl(snts: List[Dict], mode: str, file: str) -> None:
    if mode == "w":
        assert not os.path.exists(file), f"{file} exists"

    with open(file, mode, encoding="utf-8") as writer:
        for snt in snts:
            writer.write(json.dumps(snt, ensure_ascii=False)+"\n")
    print(f"writing {len(snts)} sentences to {file} with mode {mode}")


def filter_and_append_pseudo_sentences(
    predictfile: str,
    left_rawcorpus: str,
    labeled_datafile: str,
    lang: str,
    snts_num: int
) -> None:
    print(f"filter sentences from {predictfile} and append them to {labeled_datafile}")

    snts = jsonl_reader(predictfile, override_lang=lang)
    filtered_snts, left_snts = filtering(snts, snts_num)
    left_snts = [{"tokens": snt["tokens"], "postags": snt["postags"]} for snt in left_snts]

    writing_jsonl(filtered_snts, "a", labeled_datafile)
    writing_jsonl(left_snts, "w", left_rawcorpus)


if __name__ == '__main__':
    # prepare_predict_input(
    #     rawcorpus="./data/data2/origin/gd/gd.sorted.jsonl",
    #     outputfile="./results/base0/gd_input.jsonl",
    #     lang="en0",
    #     snt_start=0,
    #     snt_end=16000
    # )

    # filter_and_append_pseudo_sentences(
    #     predictfile="./results/base/roberta0/eva/sv_output.sub.jsonl",
    #     left_rawcorpus="./results/base/roberta0/eva/im_ex/sv.jsonl",
    #     labeled_datafile="./data/data2/train/base/im_ex/sv.jsonl",
    #     lang="sv1",
    #     snts_num=2000
    # )
