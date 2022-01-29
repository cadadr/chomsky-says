# Chomksy says...

These are some silly AI toys that take in a corpus and spit out some
(in)coherent text.  I'm feeding them text from Chomsky's work which
I extracted from PDFs I have using `pdf2txt.py` from `pdfminer.six`:

    $ pdf2txt.py -o choms.txt ~/Library/chomsky*.pdf

But they should be capable of working with any textual corpus.

# `v01/`: initial, simple model

A basic model that spits out an incoherent string of words, based on
a tutorial I found somewhere.  Link in the Python file.

Not very performant as it doesn't know how to save the model (tho should
be easy to edit to do that).

# `v02/`: second model from the TensorFlow tutorial

*Warning*: not working yet, outputs gibberish.

This model was made using the tutorial at
<https://www.tensorflow.org/text/tutorials/text_generationa>.

To use this model, first, edit `param.py` if necessary, then run
`train.py` to generate and train the model.  Finally, run `speak.py` to
generate text.
