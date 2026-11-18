# Design decisions

Why the project is built the way it is, in plain language.

## The question

**Why buses, and why 振替輸送?** When a rail line stops, operators often announce 振替輸送: other
lines, including buses, accept that line's tickets. Riders then crowd onto parallel bus routes,
boarding takes longer and buses fall behind schedule. Rail bulletins are published in real time,
so if the spillover is real, the bulletins should help forecast bus delays before they build up.

**Why forecast 30 minutes ahead in 15-minute steps?** Thirty minutes is long enough to act on (add a
bus, warn riders) and short enough that the rail disruption is still relevant. Fifteen minutes
matches how often the collector can reasonably run, and averages out single-bus noise.

## Evaluation

**Why not shuffle the data and use k-fold cross-validation?** Neighbouring 15-minute rows are
nearly copies of each other. Shuffling puts 10:00 in training and 10:15 in testing, so the model
"predicts" by remembering its neighbour and the score is far too good. In real use you only ever
have the past, so the test must also be the future.

**Why several test blocks instead of one train/test cut?** With a few weeks of data, a single test
period holds only a handful of rail incidents, and one unusual incident can decide the whole result.
I hit exactly this: one simulated seed's test period contained an incident four times larger than
anything in training. Rolling origin uses 4 successive test blocks, each forecast by models trained
only on data from before it, and reports results per block.

**Why drop the last 30 minutes before each test block from training?** A row's target is the delay
30 minutes later. Training rows just before the cutoff have targets *inside* the test block, so
keeping them would hand the model part of the answer.

**Why compare against persistence?** "The delay in 30 minutes equals the delay now" is free and hard
to beat. The original version only compared against a model that didn't know the current delay,
which persistence already beats. Any claimed gain has to be measured against something sensible.

**Why six models that each add one thing?** It isolates what each information source is worth. The
key comparisons are C → D (does the rail status header help?) and D → E (does reading the *text* add
anything beyond the header?). If the text adds nothing, the NLP isn't worth it, and that's a valid
finding.

**Why predict the change from now rather than the delay itself?** Tree models can only output values
in the range they saw during training. A rare, huge disruption then gets capped at "a typical bad
day". Predicting the change and adding it to the current delay anchors the forecast on what is
happening now. In the 5-seed synthetic test this more than tripled how much of the planted effect
the models recovered.

## Synthetic data

**Why simulate at all?** To test the pipeline on data where the right answer is known before trusting
it on data where it isn't. A simulation result says the code works; it says nothing about Tokyo.

**Why must the simulator never use the parser?** The first version generated bus delays *from the
parser's own severity score*, then "discovered" that the parser's score predicts bus delays. That is
circular. Now delays come from hidden incident attributes, and the text is written to describe them.
A test swaps in a nonsense parser and checks the bus delays don't change.

**Why a placebo?** If rail features "help" in a world where rail has no effect, information is
leaking from the future somewhere. The placebo world is identical except for the effect, so any
gain there is a red flag. Across 5 seeds the placebo gains stay within ±3.3%.

## Data

**How is a bus's delay measured?** Toei's feed gives each bus's position, its trip and the stop it is
heading to. It does not say how late the bus is. The bus has left the previous stop but not reached the
next one, so its delay is measured against the midpoint of the two scheduled times. That is accurate
to about half the gap between stops, roughly ±1 minute. Positions older than 5 minutes are ignored,
and buses at their first stop are skipped because they haven't departed yet.

**Why is the parser rules, not a trained model?** There are no labelled Japanese rail bulletins to
train on, and rules are transparent: every output traces to the words that caused it. Their weak spots
are known (negation, past tense, words inside station names) and are handled and tested. Real accuracy
is measured against hand labels with `src/parser_eval.py`. Labels are written from the text alone,
without looking at the parser's answer first.

**Why keep the status header and the text separate?** The header is a short controlled vocabulary
(運転見合わせ, 遅延 and so on). It's easy structured data, and sometimes missing. Keeping them apart
shows whether reading the free text adds anything beyond the header.

**Why these SQL choices?**
- The rolling hour uses `RANGE BETWEEN 2700 PRECEDING`. A `ROWS`-based window would silently stretch
  across missing buckets (buses don't run at night; collection has gaps).
- `LAG`/`LEAD` return the neighbouring *row*. That row is only 15 or 30 minutes away if nothing is
  missing, so each one checks the timestamp and returns NULL otherwise.
- `LEAD` builds the target and nothing else.
- Weather is joined from the last hour that was complete when the forecast is made. Joining the
  bucket's own hour would use readings from later in that hour. A test scrambles all data after a
  cutoff and checks that no earlier feature changes; putting the old weather join back makes it fail.

**Why SQLite only?** One file, no server, and it's enough for this data size. The original claimed
PostgreSQL support but couldn't actually run on it.

**Why commit snapshots to a git branch?** It's a free, durable store that needs no database server
("git scraping"). A separate `data` branch keeps the main branch's history clean. Each snapshot is
a new file of about 18 KB, so the repository grows by roughly 50 MB a month.

**Why are the route–rail links hand-made?** They are a hypothesis read off each route's real stop
list: which lines it runs alongside or crosses. The real-data analysis is what tests them.

## What I'd do next

- An event study: average bus delay in the hours around each real disruption, compared with the same
  route at the same time on normal days.
- A "minutes since the disruption began" feature. Long suspensions behave differently from short ones.
- Derive the route–rail links from geography (stations within a few hundred metres of a route's stops)
  instead of by hand.
- Confidence intervals per test block (bootstrap over days), so a gain can be told apart from noise.
