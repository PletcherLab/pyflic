---
status: accepted
---

# A Simulated Yoked Breaking Point replays the program on the Yoked fly's own contact

ADR-0014 made the Breaking Point the Paired fly's alone and rejected a yoked
breaking point. A Paired fly's Breaking Point mixes two things, though: the
motivation to earn the light, and any general rise in feeding that the
stimulation itself causes. In the first optogenetic PR dataset with yoked
controls, the Jove males of November 2026, yoked Chr flies made about 2.5
times as much sucrose contact as yoked W flies, so the question "would
background feeding alone have earned this breaking point?" was a real one. We
answer it with an opt-in **Simulated Yoked Breaking Point**. The group's ratio
Paradigm is replayed on the Yoked fly's own Sucrose Well signal, as if that
well had been the Trigger Well, and the simulated lights are scored by the
first-gap rule exactly as the Paired fly's real lights are.

## Decisions

- **The Yoked fly's own contact, the program's own schedule.** The schedule
  is the Test Paradigm from `Program.txt`. Its threshold is the Paired
  Trigger Well's level added to the Yoked Sucrose Well's own Baseline in
  force; its Requirement series, Activation length and contact rules are the
  firmware's (`LEDControl.c`, `StepRatio`). This differs from the yoked
  breaking point ADR-0014 rejected, which scored the Yoked fly's licks
  against requirements its partner met.
- **It starts at the group's training end**, the Paired fly's real one, with
  the first Requirement and no Progress. It is scored over the group's own
  Test window. Both flies are then measured over the same minutes of the same
  day, so their difference isolates contingency. The Yoked fly never trained,
  so no Training is simulated.
- **Contact from 5 Hz records is calibrated per Member.** The firmware
  counts contact every millisecond, but a record is one 200-ms sample. Each
  record is credited the mean contact the firmware recorded (`Progress`) on
  the Member's Paired Trigger Wells in records with the same pattern: whether
  the previous record, this one and the next were above threshold. Records
  inside a light are left out of the fit.
- **Every run checks itself.** The same replay runs on each Paired fly from
  its training end, and its breaking point (`ReplayBreakingPoint`) sits
  beside the real one. `summary.txt` reports how many groups agree within
  one light. On the Jove males, the 34 groups agreed exactly in 25 and within
  one light in 32.
- **Generation 3, a ratio Paradigm counting Contact, and a recorded Baseline
  in force.** Anything else leaves the columns NA, with one note saying why.
  An Events counter is out of scope: a 100-ms Event gap cannot be resolved
  from 200-ms records, so it would need its own calibration and check.
- **Opt-in, `pr_simulated_yoked_bp`, default false.** It is a new kind of
  number, and an analysis that does not ask for it is unchanged.
- **`dBreakingPoint` is the Breaking Point minus the Simulated Yoked Breaking
  Point**, the contingent part of the Paired fly's number. Like the other
  Paired-Yoked Differences, it is tested against zero within each treatment
  and compared between treatments. The Simulated Yoked Breaking Point is
  compared between treatments like the Breaking Point, log-rank included.

## Considered options

- **The Yoked fly's own simulated Training.** This is closer to "what if it
  had been the paired fly", and it was the first analysis, done outside
  pyflic. Not chosen: its Test then starts at a different time of day (a
  median of three hours later in the Jove males), so the comparison mixes
  contingency with circadian feeding.
- **A training-light hold of 300 ms.** At 5 Hz this matched the firmware's
  training end slightly more often than the real 500 ms. It is moot now that
  no Training is simulated: the Test light's length is read from the program.
- **Comparing feeding metrics only, as ADR-0014 does** (`dLicksA`,
  `dPersistA`). These show that the Yoked fly feeds more, but not how much of
  a breaking point that feeding is worth on this schedule. Kept alongside,
  not replaced.

## Consequences

- When the constant is on, `pr_breaking_point.csv` gains
  `SimulatedYokedBreakingPoint`, `SimulatedYokedBreakMin`,
  `SimulatedYokedCensored`, `dBreakingPoint` and `ReplayBreakingPoint`. They
  are NA for a group whose Yoked chamber is out of the analysis.
- A new `pr_simulated_yoked_lights.csv` holds every simulated Test light:
  its minute, the Requirement it met, whether it is lick-free, and whether
  it counted.
- `summary.txt`, the Combined Analysis Stats text and the Project Report
  carry the Simulated Yoked Breaking Point and `dBreakingPoint`, with the
  replay check.
- The Breaking Point itself, and every number of an analysis with the
  constant off, are unchanged.
