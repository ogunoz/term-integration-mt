# term-integration-mt
The source code repository for the paper ["Towards Precise Lexicon Integration in Neural Machine Translation"](https://aclanthology.org/2021.ranlp-1.122/)

## Usage

Providing Mullti-choice Lexical Constraints

* Instead of creating:

```bash
{ 'text': 'This is a test .', 'constraints': ['constr@@ aint', 
                                              'multi@@ word constr@@ aint',
                                             ] }
```

* Create:

```bash
{ 'text': 'This is a test .', 'constraints': [ ['constr@@ aint', 'constr@@ ain@@ ts', 'Cons@@ tr@@ aint'], 
                                               ['multi@@ word constr@@ aint', 'Multi@@ word constr@@ aint'], 
                                             ] }
```

* And let your trained NMT model to decide for the best fitting constraint options

## Benchmarks

### EN -> RU newstest2017

Model                                                                                            | Term.&nbsp;Rate | Term.&nbsp;Prec. | Term.&nbsp;Recall | Term.&nbsp;F1 | BLEU&nbsp;(Δ)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
------------------------------------------------------------------------------------------------ | :--------: | :---------: | :----------: | -------: | ------:
baseline <sub>(no constraints)</sub>                                                             | 57.43      | 78.20       | 81.16        | 79.65    | **33.2**
Dinu et al., 2019 <sub>(with all lemmata matching factors)</sub>                                 | 81.22      | 62.76       | 95.54        | 75.76    | 30.2 (-3.0)
Dinu et al., 2019 <sub>(with **BERT** selected factors)</sub>                                    | 57.17      | 79.13       | 81.45        | 80.27    | 31.8 (-1.4)
Post and Vilar, 2018 <sub>(with all lemmata matching constraints)</sub>                          | 99.88      | 49.04       | 99.23        | 65.64    | 26.0 (-7.2)
Multi-Choice Lexical Constraints <sub>(with alll lemmata matching constraints)</sub>             | 99.68      | 50.82       | 99.54        | 67.29    | 28.2 (-5.0)
Post and Vilar, 2018 <sub>(with **BERT** selected constraints)</sub>                             | 61.67      | 75.02       | 87.30        | 80.69    | 31.1 (-2.1)
Multi-Choice Lexical Constraints <sub>(with random subset of lemmata matching constraints)</sub> | 70.71      | 66.92       | 86.55        | 75.48    | 31.7 (-1.5)
**Multi-Choice Lexical Constraints*** <sub>(with **BERT** selected constraints)</sub>            | 61.62      | 77.35       | 87.30        | **82.03**| **32.5 (-0.7)**

### EN -> RU newstest2020 (extracted from ru-en wmt20/test-ts)

Model                                                                                            | Term.&nbsp;Rate | Term.&nbsp;Prec. | Term.&nbsp;Recall | Term.&nbsp;F1 | BLEU&nbsp;(Δ)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
------------------------------------------------------------------------------------------------ | :--------: | :---------: | :----------: | -------: | ------:
baseline <sub>(no constraints)</sub>                                                             | 57.33      | 77.19       | 75.01        | 76.08    | **28.8**
Dinu et al., 2019 <sub>(with all lemmata matching factors)</sub>                                 | 81.42      | 64.72       | 92.72        | 76.23    | 26.4 (-2.4)
Dinu et al., 2019 <sub>(with **BERT** selected factors)</sub>                                    | 58.27      | 79.09       | 77.88        | 78.48    | 27.8 (-1.0)
Post and Vilar, 2018 <sub>(with all lemmata matching constraints)</sub>                          | 99.79      | 51.13       | 99.32        | 67.51    | 24.6 (-4.2)
Multi-Choice Lexical Constraints <sub>(with alll lemmata matching constraints)</sub>             | 99.51      | 52.46       | 99.15        | 68.62    | 24.9 (-3.9)
Post and Vilar, 2018 <sub>(with **BERT** selected constraints)</sub>                             | 63.90      | 74.35      | 84.73        | 79.20    | 27.4 (-1.4)
Multi-Choice Lexical Constraints <sub>(with random subset of lemmata matching constraints)</sub> | 72.31      | 65.17       | 82.54        | 72.83    | 27.3 (-1.5)
**Multi-Choice Lexical Constraints*** <sub>(with **BERT** selected constraints)</sub>            | 63.84      | 75.84       | 84.52        | **79.94**| **28.1 (-0.7)**
