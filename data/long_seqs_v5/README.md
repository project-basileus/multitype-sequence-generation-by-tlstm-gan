Sequence length T = 20;

## Token Types (Token-Encoding-meaning): 

P-0-padding

N-1-initial token

A-2-start

B-3-view

C-4-click

D-5-install

## Rules

In order to be positive, a temporal sequence MUST NOT violate any of the following rules / hidden patterns:

0. __[Increasing time]__: Timestamp must be strictly increasing. A later event must have a greater timestamp then any previous event. Since delta_t is used for data generation, timestamp will always increasing, so __this rule is NOT used for oracle__

1. __[Starting with A]__: A sequence must start with an A event.

2. __[Not Only A]__: There must be a non-A token after the init token.

3. __[Pairing C & D]__: Each C event can either appear alone, or be paired with one and only one later D event. Each D event has to be paired with one and only one previous C event. Pairing can be non-unique. 

4. __[Number Decay]__: The total number of A's must be greater than B; The total number of B's must be >= the nums of C; The total number of C's must be >= the nums of D.

5. __[Minimum Same Delay]__: The minimum time delay between two consecutive __same__ tokens is 5 secs

6. __[Maximum Pair Delay]__: The time delay between the pair C and D cannot be > 100 secs


## Delta Time distribution
The delta_t distribution is conditioned on the upcoming event
e.g. if the upcoming event is an A, it follows chi-square 8 distribution

'A' : lambda: np.random.chisquare(df=8)
'B' : lambda: np.random.chisquare(df=16)
'C' : lambda: np.random.chisquare(df=24)
'D' : lambda: np.random.chisquare(df=32)