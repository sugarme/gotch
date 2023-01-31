This example implements a character-level language model heavily inspired 
by [char-rnn](https://github.com/karpathy/char-rnn).
- For training, the model takes as input a text file and learns to predict the
  next character following a given sequence.
- For generation, the model uses a fixed seed, returns the next character
  distribution from which a character is randomly sampled and this is iterated
  to generate a text.

At the end of each training epoch, some sample text is generated and printed.

Any text file can be used as an input, as long as it's large enough for training.
A typical example would be the
[tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
The training text file should be stored in `data/char-rnn/input.txt`.

To run the example:

```bash
cd example/char-rnn
go run .

```


Here is an example of generated data when training on the Shakespeare dataset
after only 5 epochs.

```
rom Dogs on the gamper, as their rear and
A king your revenues Numilitul. Your hanged
like awaket from me Mucafion as even sit best.

HENRY BOLINGBROKE:
O, I know'st me not his princess

RIVERS:
Bidgened Walter, march is their beadeful,
To be full yiel successes him a brother
And treason wid ought to do the haughty likes
Keep lay issued-formoners?

LUCIO:
Nay, but he's very were I live, I have seen.

KING EDWARD IV:
Be indeed.
What? what new and past your knee, how now!

LADY GREY:
To their own lidight right.
Will you joy?

MERCUTIO:
'Tis it that dear Bianca and York; some it
But my consent. What's the king so plant! 'LYears,
My personage of thy Lady Grumio!

KATHARINA:
Reblong, my lord;
For-sun, therefore thou high'd me to me.

ESCALUS:
But they are gone.

Second Servant:
O, they do seek out the day.

CLARENCE:
Alas; for father, lords, take they be little state,
His business: and therefore fit it should
Being well seating lie out.

Second Servant:
I know no one of purpose.

CAPULET:
Make me us.
```

