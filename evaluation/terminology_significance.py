import re


def lemmatize(sent, lemma_func):
    sent = sent.replace(" ##", "")
    sent = re.sub(r"\s?@-@\s?", "-", sent)
    sent = " ".join([x if x.isupper() else x.lower() for x in sent.split()])
    return lemma_func(sent).strip()


def stemize(sent, stem_func):
    sent = sent.replace(" ##", "")
    sent = re.sub(r"\s?@-@\s?", "-", sent)
    sent = " ".join([x if x.isupper() else x.lower() for x in sent.split()])
    return stem_func(sent).strip()


def evaluate_sentence_pair(sent_pair, term_dict, term_patterns):
    src_sent, m1_sent, m2_sent, ref_sent, m1_stem_sent, m2_stem_sent, ref_stem_sent = sent_pair

    both_in_sent = 0
    m1_not_m2_in_sent = 0
    m2_not_m1_in_sent = 0
    none_in_sent = 0

    for phrase in term_dict:
        if len(phrase) >= 2:
            pattern = term_patterns[phrase]
            src_matches = pattern.finditer(src_sent)
            src_matches = [x for x in src_matches]

            num_src_matches = len(src_matches)

            if num_src_matches > 0:
                ref_search_pattern = "(" + "|".join([x[3].pattern for x in term_dict[phrase].values()]) + ")"
                ref_stem_pattern = "(" + "|".join(["\\b" + re.escape(tgt_stem_func(x[2])) + "\\b" for x in term_dict[phrase].values()]) + ")"
                ref_matches = [x for x in re.finditer(ref_search_pattern, ref_sent)]

                ref_stem_matches = [x for x in re.finditer(ref_stem_pattern, ref_stem_sent)]
                num_ref_matches = max(len(ref_stem_matches), len(ref_matches))

                num_truth = min(num_ref_matches, num_src_matches)

                only_positive_one = list(filter(lambda x: x[1] != "N", term_dict[phrase].values()))[0]
                m1_matches = [x for x in only_positive_one[3].finditer(m1_sent)]
                m2_matches = [x for x in only_positive_one[3].finditer(m2_sent)]

                m1_stem_matches = [x for x in re.finditer("\\b" + re.escape(tgt_stem_func(only_positive_one[2])) + "\\b", m1_stem_sent)]
                m2_stem_matches = [x for x in re.finditer("\\b" + re.escape(tgt_stem_func(only_positive_one[2])) + "\\b", m2_stem_sent)]

                num_m1 = max(len(m1_matches), len(m1_stem_matches))
                num_m2 = max(len(m2_matches), len(m2_stem_matches))

                num_max_match = max(num_truth, num_m1, num_m2)

                m1_incorrect = abs(num_m1 - num_truth)
                m1_correct = num_max_match - m1_incorrect
                m2_incorrect = abs(num_m2 - num_truth)
                m2_correct = num_max_match - m2_incorrect

                both = min(m1_correct, m2_correct)
                m1_not_m2 = max((m1_correct - m2_correct), 0)
                m2_not_m1 = max((m2_correct - m1_correct), 0)
                none = min(m1_incorrect, m2_incorrect)

                both_in_sent += both
                m1_not_m2_in_sent += m1_not_m2
                m2_not_m1_in_sent += m2_not_m1
                none_in_sent += none

    return both_in_sent, m1_not_m2_in_sent, m2_not_m1_in_sent, none_in_sent


if __name__ == '__main__':

    import pickle
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
        stem_sent = " ".join([tgt_stemmer.stem(x) for x in words])
        return stem_sent.strip()


    # term_dict, term_patterns = load_group_term_dict(src_lang_gt, src_lemma_func, tgt_lang_gt, tgt_lemma_func)
    with open("../news_term_dict_no_stopwords.pickle", "rb") as handle:
        term_dict = pickle.load(handle)

    with open("../news_term_search_patterns_no_stopwords.pickle", "rb") as handle:
        term_patterns = pickle.load(handle)

    src_file = "../../test_files/newstest2020/newstest2020_extracted_moses_tokenized.en"
    model_1_output_file = "../../test_files/newstest2020/translations/mlc_bert/translations_capitalized.ru"
    model_2_output_file = "../../test_files/newstest2020/translations/lc_bert/translations_capitalized.ru"
    ref_file = "../../test_files/newstest2020/newstest2020_extracted_moses_tokenized.ru"
    both = 0
    m1_not_m2 = 0
    m2_not_m1 = 0
    none = 0

    with open(src_file, "r") as fp_src, open(ref_file, "r") as fp_ref, \
            open(model_1_output_file, "r") as fp_model_1, open(model_2_output_file, "r") as fp_model_2:

        while True:

            src_sent = fp_src.readline().strip()
            ref_sent = fp_ref.readline().strip()
            model_1_sent = fp_model_1.readline().strip()
            model_2_sent = fp_model_2.readline().strip()

            if not src_sent and not ref_sent and not model_1_sent and not model_2_sent:
                break

            sent_pair = (lemmatize(src_sent, src_lemma_func),
                         lemmatize(model_1_sent, tgt_lemma_func),
                         lemmatize(model_2_sent, tgt_lemma_func),
                         lemmatize(ref_sent, tgt_lemma_func),
                         stemize(model_1_sent, tgt_stem_func),
                         stemize(model_2_sent, tgt_stem_func),
                         stemize(ref_sent, tgt_stem_func))
            both_in_s, m1_in_s, m2_in_s, none_in_s = evaluate_sentence_pair(sent_pair, term_dict, term_patterns)

            both += both_in_s
            m1_not_m2 += m1_in_s
            m2_not_m1 += m2_in_s
            none += none_in_s

    print(both)
    print(m1_not_m2)
    print(m2_not_m1)
    print(none)

    from mlxtend.evaluate import mcnemar
    import numpy as np

    confusion = np.array([[both, m1_not_m2], [m2_not_m1, none]])
    chi2, p = mcnemar(ary=confusion, exact=False)
    print("P value: %.10f" % p)
