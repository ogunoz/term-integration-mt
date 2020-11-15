import pandas as pd
import re

def get_distinct_matches(matches):
    src_assigned_seats = set()
    distinct_matches = []
    for match in matches:
        cur_indices = set(range(match[1], match[2] + 1))
        if not bool(src_assigned_seats.intersection(cur_indices)):  # lets use this match
            # distinct_matches.append((match[0], match[3]))
            distinct_matches.append(match)
            src_assigned_seats.update(cur_indices)

    return distinct_matches


def precision_recall_for_sentence_pair(sentence_pair, term_dict, patterns, tgt_stem_func):
    src_sent = sentence_pair[0]
    trans_sent = sentence_pair[1]
    ref_sent = sentence_pair[2]
    trans_stem_sent = sentence_pair[3]
    ref_stem_sent = sentence_pair[4]

    precision_numerators = 0
    precision_denumerators = 0
    recall_numerators = 0
    recall_denumerators = 0
    
    fp_phrases = []
    tp_phrases = []

    for phrase in term_dict:
        if len(phrase) >= 2:
            pattern = patterns[phrase]
            src_matches = pattern.finditer(src_sent)
            src_matches = [x for x in src_matches]

            num_src_matches = len(src_matches)
            if num_src_matches > 0:
                ref_search_pattern = "(" + "|".join([x[3].pattern for x in term_dict[phrase].values()]) + ")"
                ref_stem_pattern = "(" + "|".join(["\\b" + re.escape(tgt_stem_func(x[2])) + "\\b" for x in term_dict[phrase].values()]) + ")"
                ref_matches = [x for x in re.finditer(ref_search_pattern, ref_sent)]
                ref_stem_matches = [x for x in re.finditer(ref_stem_pattern, ref_stem_sent)]
                num_ref_matches = max(len(ref_matches), len(ref_stem_matches))

                num_truth = min(num_ref_matches, num_src_matches)

                only_positive_one = list(filter(lambda x: x[1] != "N", term_dict[phrase].values()))[0]
                trans_matches = [x for x in only_positive_one[3].finditer(trans_sent)]
                trans_stem_matches = [x for x in re.finditer("\\b" + re.escape(tgt_stem_func(only_positive_one[2])) + "\\b", trans_stem_sent)]

                num_trans_matches = max(len(trans_matches), len(trans_stem_matches))

                precision_numerator = min(num_trans_matches, num_truth)
                recall_denumerator = num_truth

                recall_numerator = min(num_trans_matches, recall_denumerator)

                precision_numerators += precision_numerator
                precision_denumerators += num_trans_matches
                
                if precision_numerator < num_trans_matches:
                    fp_phrases.append(phrase)
                recall_numerators += recall_numerator
                recall_denumerators += recall_denumerator

    return precision_numerators, precision_denumerators, recall_numerators, recall_denumerators


def evaluate_sentence_pair(sentence_pair, term_dict, patterns, target_stem_func):
    src_sent = sentence_pair[0]
    trans_sent = sentence_pair[1]
    ref_sent = sentence_pair[2]
    trans_stem_sent = sentence_pair[3]
    ref_stem_sent = sentence_pair[4]

    trans_matches = []
    ref_matches = []

    for phrase in term_dict:
        if len(phrase) >= 2:
            pattern = patterns[phrase]
            src_match = pattern.search(src_sent)

            # src_matches = pattern.finditer(src_sent)
            # src_matches = [x for x in src_matches]
            # num_src_matches = len(src_matches)

            if src_match:
                trans_tgt_terms = list(
                    filter(lambda x: x[1] != "N" and (x[0] in trans_sent or target_stem_func(x[0]) in trans_stem_sent),
                           term_dict[phrase].values()))
                trans_matches.append((phrase, src_match.start(), src_match.end(), trans_tgt_terms))

                ref_tgt_terms = list(
                    filter(lambda x: x[1] != "N" and (x[0] in ref_sent or target_stem_func(x[0]) in ref_stem_sent),
                           term_dict[phrase].values()))
                ref_matches.append((phrase, src_match.start(), src_match.end(), ref_tgt_terms))

    # Sort found matches and
    trans_matches = sorted(trans_matches, key=lambda x: x[2] - x[1], reverse=True)
    ref_matches = sorted(ref_matches, key=lambda x: x[2] - x[1], reverse=True)

    # Get only distinct ones
    trans_matches = get_distinct_matches(trans_matches)
    ref_matches = get_distinct_matches(ref_matches)

    # Write distinct matches into dict (optional step)
    gt_trans_dict = dict()
    for match in trans_matches:
        tgt_side_str_matches = list(set(map(lambda x: x[0] + " (" + x[1] + ")", match[3])))
        # tgt_side_str_matches = list(map(lambda x: x[0] + " (" + x[1] + ")", match[3]))
        gt_trans_dict[match[0]] = " - ".join(tgt_side_str_matches)

    gt_ref_dict = dict()
    for match in ref_matches:
        tgt_side_str_matches = list(set(map(lambda x: x[0] + " (" + x[1] + ")", match[3])))
        gt_ref_dict[match[0]] = " - ".join(tgt_side_str_matches)

    return gt_trans_dict, gt_ref_dict


def evaluate(src_file, src_lemma_func, trans_file, ref_file, tgt_lemma_func, tgt_stem_func, term_dict, term_patterns,
             inplace_annotations=False, subword_prefix=" ##"):
    total_terms = 0
    total_trans_usages = 0
    total_ref_usages = 0
    gt_trans_dicts = list()
    gt_ref_dicts = list()

    precision_numerators = 0
    precision_denumerators = 0
    recall_numerators = 0
    recall_denumerators = 0

    with open(src_file, "r") as src_fp, open(trans_file, "r") as trans_fp, open(ref_file, "r") as ref_fp:

        while True:

            src_sentence = src_fp.readline().strip()
            trans_sentence = trans_fp.readline().strip()
            ref_sentence = ref_fp.readline().strip()

            if not src_sentence and not trans_sentence and not ref_sentence:
                break

            real_src_sent = src_sentence
            if inplace_annotations:
                real_src_tokens = [x[:-2] for x in src_sentence.strip().split()]
                real_src_sent = " ".join(real_src_tokens)

            real_src_sent = real_src_sent.replace(subword_prefix, "")
            real_src_sent = re.sub(r"\s?@-@\s?", "-", real_src_sent)
            src_tokens = [x if x.isupper() else x.lower() for x in real_src_sent.strip().split()]
            src_sent = src_lemma_func(" ".join(src_tokens).strip()).strip()

            real_trans_sent = trans_sentence.replace(subword_prefix, "")
            real_trans_sent = re.sub(r"\s?@-@\s?", "-", real_trans_sent)
            trans_tokens = [x if x.isupper() else x.lower().strip() for x in real_trans_sent.strip().split()]
            trans_sent = tgt_lemma_func(" ".join(trans_tokens).strip()).strip()

            trans_stem_sent = tgt_stem_func(" ".join(trans_tokens).strip())

            real_ref_sent = ref_sentence.replace(subword_prefix, "")
            real_ref_sent = re.sub(r"\s?@-@\s?", "-", real_ref_sent)
            ref_tokens = [x if x.isupper() else x.lower().strip() for x in real_ref_sent.strip().split()]
            ref_sent = tgt_lemma_func(" ".join(ref_tokens).strip()).strip()

            ref_stem_sent = tgt_stem_func(" ".join(ref_tokens).strip())

            gt_dicts_of_sent = evaluate_sentence_pair((src_sent.strip(), trans_sent.strip(), ref_sent.strip(),
                                                       trans_stem_sent.strip(), ref_stem_sent.strip()),
                                                      term_dict,
                                                      term_patterns,
                                                      tgt_stem_func)

            prec_num, prec_de, recall_num, recall_de = precision_recall_for_sentence_pair(
                (src_sent.strip(), trans_sent.strip(), ref_sent.strip(),
                 trans_stem_sent.strip(), ref_stem_sent.strip()),
                term_dict,
                term_patterns,
                tgt_stem_func)

            precision_numerators += prec_num
            precision_denumerators += prec_de
            recall_numerators += recall_num
            recall_denumerators += recall_de

            gt_trans_dicts.append(gt_dicts_of_sent[0])
            gt_ref_dicts.append(gt_dicts_of_sent[1])
            total_trans_usages += len(list(filter(lambda x: x, gt_dicts_of_sent[0].values())))
            total_ref_usages += len(list(filter(lambda x: x, gt_dicts_of_sent[1].values())))
            total_terms += len(set(list(gt_dicts_of_sent[0].keys()) + list(gt_dicts_of_sent[1].keys())))

    print("Translation Term Usage: %.4f%%" % (100.0 * (total_trans_usages / total_terms)))
    print("Reference Term Usage: %.4f%%" % (100.0 * (total_ref_usages / total_terms)))
    print()
    print("Total matched terms in translations: %d" % total_trans_usages)
    print("Total matched terms in references: %d" % total_ref_usages)
    print("Total source terms: %d" % total_terms)

    print("Precision: %d/%d" % (precision_numerators, precision_denumerators))
    print("Recall: %d/%d" % (recall_numerators, recall_denumerators))
    precision_f = precision_numerators / precision_denumerators
    recall_f = recall_numerators / recall_denumerators
    f1 = 2 * (precision_f * recall_f) / (precision_f + recall_f)
    print("F1 score: %f " % f1)

    df = pd.DataFrame.from_dict({"Group Terms in Translation": gt_trans_dicts,
                                 "Group Terms in Reference": gt_ref_dicts})
    return df


if __name__ == '__main__':
    import spacy
    from pymystem3 import Mystem

    from nltk.stem.snowball import RussianStemmer

    tgt_stemmer = RussianStemmer()

    from spacy.tokens import Doc

    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])


    def src_lemma_func(text):
        words = text.split()
        spaces = [True] * len(words)
        doc = Doc(nlp.vocab, words=words, spaces=spaces)

        term_tokens = [x.lemma_ for x in doc]
        sent = " ".join(term_tokens)
        return sent.strip()


    ru_lemmatizer = Mystem()


    def tgt_lemma_func(text):
        tgt_tokens = ru_lemmatizer.lemmatize(text)
        sent = "".join(tgt_tokens)
        return sent.strip()


    def tgt_stem_func(text):
        words = text.strip().split()
        return " ".join([tgt_stemmer.stem(x) for x in words]).strip()


    src_test_file = "../../test_files/newstest2020/newstest2020_extracted_moses_tokenized.en"
    translation_file = "../../test_files/newstest2020/translations/mlc_bert/translations_capitalized.ru"
    tgt_ref_file = "../../test_files/newstest2020/newstest2020_extracted_moses_tokenized.ru"
    inplace_annotations = False

    # Load Group Term Dict
    import pickle

    with open("../news_term_dict_no_stopwords.pickle", "rb") as handle:
        term_dict = pickle.load(handle)

    with open("../news_term_search_patterns_no_stopwords.pickle", "rb") as handle:
        term_patterns = pickle.load(handle)

    df = evaluate(src_file=src_test_file,
                  src_lemma_func=src_lemma_func,
                  trans_file=translation_file,
                  ref_file=tgt_ref_file,
                  tgt_lemma_func=tgt_lemma_func,
                  tgt_stem_func=tgt_stem_func,
                  term_dict=term_dict,
                  term_patterns=term_patterns,
                  inplace_annotations=False)

    print(df)
