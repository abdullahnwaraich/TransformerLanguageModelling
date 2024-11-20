# TransformerLanguageModelling
This repository contains necessary files to complete Assignment 3 of CS388 UT-Austin (https://www.cs.utexas.edu/~gdurrett/courses/online-course/a3.pdf).
A general transformer architecture is implemented from first principles in transformer.py. Training procedure for language modelling are given in 
transformer_lm.py and lm.py. Although constrained by modest computational resources, with merely two attention heads and embedding layers, in ten epochs
trained model achieves a perplexity of <5 on dev_text: 
=====Results=====
{
  "sane": true,
  "normalizes": true,
  "range": true,
  "log_prob": -718.0046058912776,
  "avg_log_prob": -1.4360092117825551,
  "perplexity": 4.203885479419891
}  
Decoding routines like beam_search and nucleus sampling are provided in lm.py.
