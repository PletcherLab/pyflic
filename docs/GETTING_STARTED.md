# Getting Started with pyflic

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
| Open the hub on a specific folder | `uv run pyflic hub "path/to/my_project"` |
| Open the config editor | `uv run pyflic config` |
| Check a config for mistakes | `uv run pyflic lint "path/to/my_experiment"` |
| Make a PDF report | `uv run pyflic report "path/to/my_project"` |
| Open the built-in help | `uv run pyflic help` |
| List all commands | `uv run pyflic --help` |

Put folder paths in double quotes, especially if they contain spaces.

> **Finding a folder's path.** You can drag a folder from File Explorer (Windows) or
> Finder (macOS) into the terminal window, and its full path is typed in for you.

See the [README](../README.md) for what each command does and how to lay out your
experiment folders.

---

## Everyday use

Once setup is done, each time you want to use pyflic:

1. Open a terminal ([Step 1](#step-1-open-a-terminal)).
2. Go to the pyflic folder:
   - Windows: `cd ~\Documents\pyflic`
   - macOS / Linux: `cd ~/Documents/pyflic`
3. Start pyflic: `uv run pyflic`

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
