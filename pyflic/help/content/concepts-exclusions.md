# Excluding chambers in bulk

An **excluded chamber** is removed from every result — it never reaches a summary, a plot,
or a statistic. Each member declares its own in a `remove_chambers.csv` at its root:

```csv
group,dfm_id,chamber,note
general,1,3,low lick count
general,2,5,fly escaped
```

The `group` column lets one member carry several exclusion sets; the Project's Design
names the active one with `exclusion_group:` (default `general`), so every member is
filtered by the same rule. Each [Slice](concepts-slices.md#exclusions) has its own group, and
`also_exclude:` names groups an analysis excludes as well, such as one for flies dead from
the start. That file is the audit trail, and everything below only ever
writes into it — as does the QC Viewer's **Save Exclusions…**, for one member at a time
(see [Saving exclusions](app-qc-viewer.md#saving-exclusions)).

## The problem this solves

You watched forty recordings and made notes on a dozen chambers. Declaring them means
opening forty directories and editing forty files, one row at a time. Work that tedious
does not get done, and chambers that should have been excluded quietly stay in.

## The exclusion sheet

Put a **`remove_chambers.csv` at your batch folder or your project folder** — one level
*above* the per-member files — and list everything at once:

```csv
project,member,dfm,chamber,group,reason
Sept2026/ProjA,Feb_dose_low,1,3,general,low lick count
Sept2026/ProjA,Feb_dose_high,2,5,general,fly escaped
Archive/2025/ProjC,pilot,1,1,general,noisy signal
```

`.xlsx` (or `.xls`) works too; when both are present the `.csv` is read. Column names are matched loosely — `Experiment` or `Replicate` are
accepted for `member`, `Note` for `reason` — and `project`, `group` and `reason` are all
optional. A sheet at a **project** root needs no `project` column at all. `project` uses
the same path-shaped name the Batch table shows.

## What applying it does

It **writes the rows down into each member's own `remove_chambers.csv`**. Nothing reads the
sheet at analysis time. That is the whole design: a sheet you delete, move, or never apply
changes no result, and every exclusion that reaches an analysis is visible in the file the
analysis already reports from.

Three rules keep it from becoming a second source of truth:

- **The standing declaration wins.** A chamber already declared is never rewritten. If the
  sheet gives a different reason, that is reported as a **conflict** and the existing
  reason is kept — a Batch Run re-applies the sheet every time, so letting the sheet win
  would keep overwriting reasons you refined by hand. Edit the member's own file to change
  one.
- **Selecting reports; it never applies.** Opening a batch folder tells you a sheet is
  there and how many rows it has. Writing happens only when you press **Apply exclusion
  sheet…** or start a Batch Run.
- **Only the projects that are running are touched.** A row naming an unchecked project is
  skipped and reported. Unchecking a project means "do not touch this project".

A Batch Run's review window previews every row's outcome and offers one switch to decline
the sheet for that run — without editing the sheet or any declaration.

## What each row can say

| Outcome | Meaning |
|---|---|
| `applied` | written into that member's file |
| `already declared` | that chamber is already excluded with the same reason |
| `conflict` | already excluded with a *different* reason; the existing one is kept |
| `unknown project` / `unknown member` | no such directory under the sheet's folder |
| `unknown chamber` | that DFM is not configured in that member |
| `incomplete` | `member`, `dfm` and `chamber` are required, and dfm/chamber must be whole numbers |

Nothing here is fatal. An unreadable sheet is reported and the run continues; one
unwritable member directory does not discard the others.

## Results that predate a declaration

Declaring a chamber does nothing to results already on disk. When a member's
`remove_chambers.csv` is newer than its saved `analysis/feeding_summary.csv`, the Hub's
Analyzed column reads **re-run needed** rather than *yes*: those numbers describe a chamber
population you have since said was wrong, and a date beside them would be true about a
number that is not.

Re-run the member's analysis to clear it.

## Automatic exclusions are separate

`auto_remove_chambers()` removes chambers that fail the `constants:` cutoffs (the
Design's, in a Project):

- too few licks to be a fly at all (`min_untransformed_licks_cutoff`, on untransformed
  licks);
- in a two-well experiment, an implausible median bout duration or event count at either
  well (`max_med_duration_cutoff`, `max_events_cutoff`);
- a well resting at the [signal ceiling](config-structure.md#the-signal-ceiling), where no
  lick can show, for more than `max_ceiling_minutes` (30 by default, every type) of the
  analysed range;
- in an optogenetic experiment with `exclude_failed_opto_chambers` on (it is off by
  default), every chamber of a linkage group that failed the light QC;
- in a Progressive Ratio experiment, a chamber group whose training never finished
  (`require_training_complete`) or whose light the fly did not earn
  (`exclude_failed_pr_groups`).

Each of the numeric cutoffs is optional: YAML `null` skips that check, and so does an
unset key for the three a type may leave without a default (the ceiling's is 30
everywhere); only NaN lick values are always removed. Basic analysis, and the Hub's **Run analysis**, run
auto-removal once per member, before writing the summary, and its reasons are in the
member's `analysis/removed_chambers.csv`, in `summary.txt`, and in the Project Report's
exclusions table alongside the declared ones. The sheet has nothing to do with it: one is
a machine verdict, the other is the experimenter's.
