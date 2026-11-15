# Getting Started with pyflic

> **pyflic is under active development.** Its features, screens and installation steps
> change from time to time, so some details in this guide may be out of date. The latest
> code and documentation are always on pyflic's GitHub page:
> <https://github.com/PletcherLab/pyflic>. To move to the newest version, see
> [Updating pyflic](#updating-pyflic).

This guide takes you from nothing installed to pyflic running on your computer. It
assumes no experience with Python, GitHub, or the command line. Setup takes about
15 minutes, most of it waiting for downloads.

It covers **Windows**, **macOS**, and **Linux**. Where the steps differ, each system
gets its own section. Follow the one for your computer and skip the others.

---

## What you are about to do

You will do these things once:

1. **Open a terminal.** A terminal is a window where you type commands instead of
   clicking buttons. You will only need a handful of short commands, and this guide
   gives you each one to copy and paste.
2. **Install uv.** uv is a free tool that installs Python and every package pyflic
   needs, then runs pyflic. You do not need to install Python yourself; uv does it for
   you and keeps it separate from anything else on your computer.
3. **Download pyflic** from GitHub, the website where pyflic's code is stored. You can
   do this in one of two ways:
   - **Option A: download a ZIP file** with your web browser. Nothing extra to
     install. **Choose this if you are new to all of this.**
   - **Option B: use Git**, a tool for downloading and updating code. It takes one extra
     install, but updating pyflic later becomes a single command.
4. **Install pyflic's packages and run it.**

After that, starting pyflic takes two commands (see [Everyday use](#everyday-use)).

> **Typing commands.** In this guide, text in a grey box like `uv --version` is a
> command. Type it (or copy and paste it) into the terminal exactly as shown, then
> press **Enter**. Capital letters, spaces, and punctuation all matter.

---

## Step 1: Open a terminal

### Windows

1. Press the **Windows key**, type `PowerShell`, and press **Enter**.
2. A window with a blue or black background opens. This is your terminal.

(If you have **Windows Terminal**, it works too. It opens PowerShell by default.)

### macOS

1. Press **Command (⌘) + Space** to open Spotlight.
2. Type `Terminal` and press **Enter**.

### Linux

Press **Ctrl + Alt + T**, or look for **Terminal** in your applications menu.

---

## Step 2: Install uv

### Windows

In PowerShell, paste this command and press Enter:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS and Linux

In Terminal, paste this command and press Enter:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(On macOS, if you already use Homebrew, `brew install uv` also works.)

### All systems: check that it worked

**Close the terminal and open a new one.** This is required: an already-open terminal
won't find newly installed programs. Then type:

```
uv --version
```

You should see something like `uv 0.9.x`. If you get "not recognized" or "command not
found", see [Troubleshooting](#troubleshooting).

---

## Step 3: Download pyflic

Pick **one** option. Both end with a folder named `pyflic` inside your **Documents**
folder, so the rest of the guide is the same whichever you choose.

### Option A: Download a ZIP (no Git needed)

1. In your web browser, go to <https://github.com/PletcherLab/pyflic>.
2. Click the green **Code** button near the top of the page, then click
   **Download ZIP**. A file named `pyflic-main.zip` downloads, usually into your
   **Downloads** folder.
3. Unzip it:
   - **Windows:** open your Downloads folder in File Explorer, right-click
     `pyflic-main.zip`, and choose **Extract All…**, then **Extract**.
   - **macOS:** double-click `pyflic-main.zip` in Finder. (Safari sometimes unzips it
     for you, in which case there is already a `pyflic-main` folder.)
   - **Linux:** right-click the file and choose **Extract Here**, or run
     `unzip ~/Downloads/pyflic-main.zip -d ~/Downloads`.
4. Find the right folder. It is named `pyflic-main`, and **inside it you should see
   files such as `pyproject.toml` and `README.md`**. On Windows the unzipped folder
   often has *another* `pyflic-main` folder inside it; if so, use that inner one.
5. Move that folder into your **Documents** folder (drag it there).
6. Rename it from `pyflic-main` to **`pyflic`**. (Right-click → **Rename** on Windows
   and Linux; click the name once and press **Enter** on macOS.)

You now have a `Documents/pyflic` folder. Go on to [Step 4](#step-4-install-pyflics-packages).

### Option B: Download with Git

**First, install Git** if you don't have it. To check, type this in the terminal:

```
git --version
```

If you see something like `git version 2.45.1`, Git is already installed; skip ahead to
**Clone pyflic** below. If you see an error such as "not recognized" or "command not
found", install it:

- **Windows:** go to <https://git-scm.com/downloads/win>, download the installer
  ("64-bit Git for Windows Setup"), and run it. Clicking **Next** on every screen and
  keeping the default choices is fine. Then **close PowerShell and open a new one**.
- **macOS:** running `git --version` offers to install the **Command Line Developer
  Tools**. Click **Install** and wait for it to finish (this can take several minutes).
- **Linux:** use your distribution's package manager:
  - Ubuntu / Debian: `sudo apt update && sudo apt install git`
  - Fedora: `sudo dnf install git`
  - Arch: `sudo pacman -S git`

  `sudo` asks for your computer password. Nothing appears on screen as you type it; that
  is normal. Type it and press Enter.

Run `git --version` again to confirm it worked.

**Clone pyflic.** Move the terminal into your Documents folder:

- Windows: `cd ~\Documents`
- macOS / Linux: `cd ~/Documents`

Then download the code (this command is the same on every system):

```
git clone https://github.com/PletcherLab/pyflic.git
```

Git creates a `pyflic` folder inside Documents and downloads the code into it.
Downloading with Git is called **cloning**.

---

## Step 4: Install pyflic's packages

First, move the terminal into the `pyflic` folder:

**Windows (PowerShell):**

```powershell
cd ~\Documents\pyflic
```

**macOS / Linux:**

```bash
cd ~/Documents/pyflic
```

`cd` means "change directory" (a directory is a folder). `~` is short for your home
folder.

Then run:

```
uv sync
```

uv now downloads the right version of Python (3.13) and all the packages pyflic
depends on. **The first run takes a few minutes** and prints a long list of package
names. When the list stops and the prompt returns, it is done.

You only need to do this once. After updating pyflic, `uv run` in the next step
installs anything new automatically.

---

## Step 5: Run pyflic

From inside the `pyflic` folder:

```
uv run pyflic
```

The **pyflic analysis hub** opens in its own window. The first launch can take several
seconds.

That's it: pyflic is installed and running. 🎉

To check from the terminal, run `uv run pyflic version`, which prints the installed
version number. (If you used the ZIP download, it prints `0.0.0`. That's expected,
because a ZIP doesn't carry version information.)

### Other pyflic commands

Every pyflic command is typed as `uv run` followed by the command. For example:

| What you want | Command |
|---|---|
| Open the analysis hub | `uv run pyflic` |
| Open the hub on a recording, a Project or a folder of Projects | `uv run pyflic hub "path/to/my_project"` |
| Open the config editor on a recording | `uv run pyflic config "path/to/my_experiment"` |
| Open the Experiment Type Editor | `uv run pyflic-types` |
| Check a config for mistakes | `uv run pyflic lint "path/to/my_experiment"` |
| Check a recording's quality in the QC viewer | `uv run pyflic qc "path/to/my_experiment"` |
| Make a PDF report | `uv run pyflic report "path/to/my_project"` |
| Open the Plot Editor for a project's figures | `uv run pyflic plots "path/to/my_project"` |
| Run a project script in every project under a folder | `uv run pyflic batch "path/to/my_study"` |
| Delete pyflic's cached results for a folder | `uv run pyflic clear-cache "path/to/my_experiment"` |
| Show the installed version | `uv run pyflic version` |
| Open the built-in help | `uv run pyflic help` |
| List all commands | `uv run pyflic --help` |

Put folder paths in double quotes, especially if they contain spaces.

> **Finding a folder's path.** You can drag a folder from File Explorer (Windows) or
> Finder (macOS) into the terminal window, and its full path is typed in for you.

See the [README](../README.md) for what each command does.

---

## Next steps: your first analysis

pyflic is installed. From here on the work happens in pyflic's own windows, and the
built-in help walks you through it: press **F1** in the hub, or run `uv run pyflic help`.
Its **Guides** list starts with *Your first experiment*, which goes from an empty folder
to your first plot.

In outline:

1. **Make a folder for the recording**, outside the pyflic folder, with a `data`
   folder inside it. Copy everything the MCU exported for the recording into `data`: the
   DFM files and, when the MCU wrote them, `Program.txt`, `Messages.csv` and
   `Baselines.csv`. pyflic reads recordings from every version of PyMCU and the DFM
   firmware. `Program.txt` matters most for an optogenetic recording: pyflic uses it to
   check that the light followed the flies, and that the recording fits its Experiment
   Type. Without it, pyflic warns and leaves the light results blank.

   ```
   my_experiment/
     data/
       DFM1_0.csv
       DFM2_0.csv
       Program.txt
       Messages.csv
       Baselines.csv
   ```

2. **Describe the experiment** in the config editor. Give it the recording's folder:

   ```
   uv run pyflic config "path/to/my_experiment"
   ```

   For a folder with no config yet, it finds the DFM files in `data/` and opens a tab for
   each DFM, ready for you to assign its chambers. On the **Experiment** tab you choose the **Experiment Type** from
   its menu; see [Which Experiment Type?](#which-experiment-type) below. **File → Save**
   writes your answers as `flic_config.yaml` in the recording's folder.

   Always name the folder. Started without one, the config editor opens the
   `flic_config.yaml` of the folder the terminal is in, if it has one. Anywhere else it
   opens empty, and **File → Save** asks where to save.
3. **Open the hub on the folder:**

   ```
   uv run pyflic hub "path/to/my_experiment"
   ```

   The hub loads the recording, and the **Experiment** tile opens a second row of tiles:
   **QC · Analyze · Plots · Scripts · AI**.
4. **Check the data before you believe any result.** Click **QC**, then **Run QC**. It
   writes the integrity report, the signal plots and the other checks into the `qc/`
   folder. Then click **Open QC Viewer**, on the panel's **View** card, to look at them.
5. **Analyse.** Click **Analyze**, then **Run analysis**. It always writes the feeding
   summary and `summary.txt` into the `analysis/` folder, then each ticked analysis, and
   the experiment report (`experiment_report.pdf`) last.
6. **Draw the figures.** Click **Plots**, then **Generate plots**. Each ticked figure is
   saved into `analysis/` and opens in a tab of its own.

The QC, Analyze and Plots panels list what their button writes as checkboxes, in groups,
and every box starts ticked. Untick what you do not need: the choice is saved at once, in
the recording's `flic_config.yaml`, and the next run leaves out the same. A box that is
missing does not apply to this recording (a two-well check on a single-well layout, say),
and a greyed box is one the data cannot answer. One line under the groups names what is
not offered; hover over it, or over a greyed box, for the reason.

Several recordings that answer one question belong together in a **Project**, a folder
whose `project.yaml` holds the design they share. The hub's **Project** panel sets one up
(**Create project…**, or **Initialize existing directory…** for a folder of recordings you
already have), and you load a member by double-clicking its row. In a Project the boxes are
saved in `project.yaml` and apply to every member. The Project panel's **Analyze all** runs
every member's QC, analysis and report in one go, whatever the boxes say.

### Which Experiment Type?

The Experiment Type is the assay. It fixes the chamber layout where the assay needs one,
names the wells that must be labelled, decides which analyses and figures the hub offers,
and states the light the recording should have: a recording whose `Program.txt` runs
other light fails to load, with the Paradigm named. Whatever the type, pyflic judges the
light against each Paradigm's own rule and reports its own measures.

pyflic ships three types. The config editor's menu names them as below, with the name
`flic_config.yaml` stores in brackets:

| Your experiment | Experiment Type |
|---|---|
| No light, or any light program you do not want to declare | **Custom Experiment** (`Custom`), with its chamber layout: single-well or two-well |
| A two-well hedonic design, where bout duration matters, without light | **Hedonic Feeding** (`Hedonic`) |
| A ratio program (ProgressiveRatio or FixedRatio Paradigms), with or without yoked controls, and no other light but open-loop intervals | **Progressive Ratio** (`ProgressiveRatio`) |
| Anything your lab runs again and again: a hedonic design under closed-loop light, ClosedLoop Training before a ratio, a genotype panel against its control | a **type of your own**, built once in the Experiment Type Editor |

A paired and yoked design under Custom, or under a type with the *Paired and yoked*
module, gets its paired − yoked comparison.

**The Experiment Type Editor** builds a type of your own. Open it with
`uv run pyflic-types`, with **Experiment type editor** on the hub's **Tools** panel, or
with **New type…** beside the Experiment Type menu. The three shipped types open
read-only; **New from…** copies one under a name of your own, as a starting point. A type
states its chamber layout, the light it expects, the wells and design factors it needs,
its **Analysis Modules** (Progressive Ratio, Hedonic, Paired and yoked, Light and dark),
and, if you want them, the **control comparisons** its reports lead with: two Roles
compared, or every level of a factor against a control level. **Save to library** keeps it
in your lab's type library (by default `~/.config/pyflic/experiment_types`; point it at a
shared drive to share it), and the menu then lists it marked *(lab)*. Saving a config that
names it writes a copy beside the config as `experiment_type.yaml`, so a colleague's
machine analyses the recording the same way. The built-in help's *Experiment types and
chamber layouts* and *Experiment Type Editor* topics cover the rest.

Two more guides sit beside this one:

| Guide | Read it when |
|---|---|
| [Checking your data](qc-user-guide.md) | you want to know what every quality-control check, table and figure means, for ordinary and optogenetic experiments alike |
| [Analysing a Progressive Ratio assay](progressive-ratio-user-guide.md) | you run Progressive Ratio experiments |

---

## Everyday use

Once setup is done, each time you want to use pyflic:

1. Open a terminal ([Step 1](#step-1-open-a-terminal)).
2. Go to the pyflic folder:
   - Windows: `cd ~\Documents\pyflic`
   - macOS / Linux: `cd ~/Documents/pyflic`
3. Start pyflic: `uv run pyflic`, or open the hub straight on your work with
   `uv run pyflic hub "path/to/my_project"`.
4. In the hub, open your Project from the **Project** panel (**Open Project**) and
   double-click a member to load it; then use its **QC**, **Analyze** and **Plots** tiles,
   as in [Next steps](#next-steps-your-first-analysis).

Keep your own experiment data **outside** the pyflic folder, for example in
`Documents/FLIC data`. Then updating pyflic can never touch your data.

### Updating pyflic

**If you used the ZIP (Option A):**

1. Delete (or rename) your `Documents/pyflic` folder. Make sure nothing of yours is
   stored inside it first.
2. Repeat [Option A](#option-a-download-a-zip-no-git-needed) to download the latest
   ZIP and set up a fresh `Documents/pyflic` folder.
3. Run `uv run pyflic` as usual. uv installs any new packages automatically.

**If you used Git (Option B):** from inside the `pyflic` folder, run

```
git pull
```

then `uv run pyflic` as usual. If `git pull` complains that your local changes would
be overwritten, you have edited files inside the pyflic folder; ask for help before
going further.

---

## Troubleshooting

**"uv" or "git" is not recognized / command not found**
Close every terminal window and open a new one. Terminals only find programs that were
installed before they opened. If it still fails on macOS or Linux, run
`source ~/.local/bin/env` (or log out and back in). On Windows, restart the computer.

**Windows: "running scripts is disabled on this system"**
This affects the uv install command. Make sure you pasted the full command from
[Step 2](#step-2-install-uv), including `-ExecutionPolicy ByPass`.

**"Cannot find path … Documents\pyflic" or "No such file or directory"**
The folder isn't where the command expects. Check that `Documents` contains a folder
named exactly `pyflic` (not `pyflic-main`, and not a `.zip` file). If you used the ZIP,
see steps 4–6 of [Option A](#option-a-download-a-zip-no-git-needed).

**"No `pyproject.toml` found" or "Failed to spawn: pyflic"**
The terminal isn't inside the right folder. `cd` into it
([Everyday use](#everyday-use), step 2) and try again. On Windows, `dir` lists the files
in the current folder; on macOS/Linux, `ls` does. You should see `pyproject.toml` and
`README.md` among them. If you only see another folder, the code is one level further
in: `cd` into that folder, or move it up as described in
[Option A](#option-a-download-a-zip-no-git-needed).

**"fatal: destination path 'pyflic' already exists"**
You have already cloned pyflic. Carry on with [Step 4](#step-4-install-pyflics-packages).

**Linux: the window doesn't open and the terminal mentions the "xcb" platform plugin**
pyflic's windows use Qt, which needs a system library that some Linux installs lack.
On Ubuntu / Debian: `sudo apt install libxcb-cursor0`. On Fedora:
`sudo dnf install xcb-util-cursor`. Then run `uv run pyflic` again.

**macOS: a security warning appears**
If macOS asks whether Terminal may access your Documents or Desktop folder, click
**Allow**.

**The config editor opened settings you did not enter**
It was started without a folder, so it opened the `flic_config.yaml` of the folder the
terminal was in. Close it without saving, and start it on your recording:
`uv run pyflic config "path/to/my_experiment"`.

**The Experiment tile is greyed, and there is no QC, Analyze or Plots tile**
No recording is loaded. Open it with `uv run pyflic hub "path/to/my_experiment"`, or, in a
Project, double-click the member's row on the **Project** panel. Once the load finishes,
the Experiment tile opens its row of tiles.

**A checkbox you expected is missing or greyed**
A missing box does not apply to this recording: a two-well check on a single-well layout,
a light check on a recording without light, or an analysis the Experiment Type does not
include. One line under the groups names what is not offered. A greyed box applies but the
data cannot answer it, such as the Program timeline without a `Program.txt`. Hover over
the line or the box for the reason.

**A run left something out, or a file in `analysis/` or `qc/` is out of date**
Its box was unticked, perhaps by someone else in the same Project. The log lists every item
a run left out. Tick the box and run again. A run never removes a file, so the copy from
an earlier run stays until a run writes it again.

**Loading fails with "does not satisfy experiment_type"**
The recording does not fit the Experiment Type its config names. When the message says
*the Opto Program runs ClosedLoop* (or another Paradigm type), the `Program.txt` runs light
that type does not expect: choose another type, or build one that expects that light in
the Experiment Type Editor ([Which Experiment Type?](#which-experiment-type)).

**Something else**
Copy the full error text from the terminal and include it when you ask for help, or
open an issue at <https://github.com/PletcherLab/pyflic/issues>.

---

## Glossary

| Term | Meaning |
|---|---|
| **Terminal** | A window where you type commands. On Windows it is PowerShell; on macOS and Linux it is Terminal. |
| **Command** | A line of text you type into the terminal and run with Enter. |
| **Folder / directory** | The same thing. Terminals usually say "directory". |
| **`cd`** | "Change directory": moves the terminal into a folder. |
| **GitHub** | The website where pyflic's code is stored. |
| **Repository (repo)** | A project's folder of code, as stored on GitHub. |
| **ZIP file** | A single compressed file holding a folder. Unzipping (extracting) it gives you the folder back. |
| **Git** | An optional tool that downloads code and keeps it up to date. |
| **Clone** | Download a repository with Git so it can be updated later. |
| **Python** | The programming language pyflic is written in. uv installs it for you. |
| **uv** | A tool that installs Python and pyflic's packages, and runs pyflic. |
| **Package** | A piece of code that pyflic depends on (for example, for plotting). |
| **Analysis Hub (hub)** | pyflic's main window, which `uv run pyflic` opens. |
| **Experiment Directory** | One recording's folder: its `flic_config.yaml`, its `data/` folder of DFM files, and the `qc/` and `analysis/` folders pyflic writes. |
| **Project** | A folder with a `project.yaml` whose Experiment Directories, its **members**, are different experiments addressing one question. The Project's design is shared by every member, and the Project pools their results. |
| **Experiment Type** | The assay a recording is analysed as. pyflic ships Custom, Hedonic and ProgressiveRatio; a lab builds its own in the Experiment Type Editor. |
