# Git workflow

The code lives on GitHub at `git@github.com:PhyAMR/Neutrions.git`. Everyone commits to `main`.

Each of you works on your own VM, in your own copy of the repository (a *clone*). **GitHub is the only place where work is shared**: if you haven't pushed it, the others can't see it.

## One-time setup

Do this on your VM. [CONNECT.md](CONNECT.md) explains how to log in.

### 1. Tell git who you are

```bash
git config --global user.name  "First Last"
git config --global user.email "you@example.com"
git config --global pull.rebase false
```

Without the first two lines your commits show up as "Your Name" or as `ubuntu`, and nobody can tell who wrote what. The last line makes `git pull` merge, which is the simplest behaviour.

### 2. Connect to GitHub

1. Create an SSH key on this machine if you don't have one: `ssh-keygen -t ed25519` (press Enter to accept the defaults).
2. Print it with `cat ~/.ssh/id_ed25519.pub` and paste that line into GitHub → **Settings** → **SSH and GPG keys** → **New SSH key**.
3. Ask the owner of the repository (`PhyAMR` on GitHub) to add you as a collaborator.
4. Test it: `ssh -T git@github.com` should greet you by your GitHub username.

### 3. Clone the repository

```bash
cd ~
git clone git@github.com:PhyAMR/Neutrions.git Grassi
cd Grassi
```

### 4. Get the data

`data/` is not in git because it is too large (about 800 MB), so you add it by hand. It goes **inside your clone**, with exactly this layout:

```text
Grassi/
├── data/
│   ├── raw/
│   │   ├── training/
│   │   │   ├── events.parquet
│   │   │   └── truth.parquet
│   │   └── test/                (empty for now)
│   └── processed/               (created by src/scripts/preprocessing.py; don't copy it)
├── src/
└── ...
```

Create the folders:

```bash
mkdir -p data/raw/training data/raw/test
```

Then get `events.parquet` and `truth.parquet` from a teammate who has them and put them in `data/raw/training/`. If the files are on your laptop, copy them to your VM with the SSH config from [CONNECT.md](CONNECT.md) (run this on your laptop):

```bash
scp events.parquet truth.parquet cloud-vm:~/Grassi/data/raw/training/
```

## Every time you work

```bash
# 1. Get everyone else's work before you start
git pull

# 2. Work, and save snapshots as you go
git status                     # what did I change?
git add src/some_file.py       # choose the files to save
git commit -m "Say in one line what changed"

# 3. Before you upload, get anything new from the others
git pull

# 4. Upload
git push
```

If `git push` says **rejected**, someone pushed while you were working. Run `git pull`, then `git push` again.

## Rules

- **Pull before you start and before you push.**
- **Push often.** Small commits pushed every few hours give fewer and smaller conflicts than one big push at the end.
- **Commit code that runs.** Everyone uses the same `main`, so if it's broken, it's broken for everyone.
- **Add only the files you meant to change.** Avoid `git add .`, because it also picks up data, outputs and temporary files.
- **Say when you're working on a shared file** (for example `src/scripts/preprocessing.py`), so two people don't edit the same lines.

## Fixing a conflict

When you and someone else changed the same lines, `git pull` can't choose between them and marks the file:

```text
<<<<<<< HEAD
your version
=======
the version from GitHub
>>>>>>> ...
```

1. Open the file. VS Code shows buttons above each conflict: **Accept Current** (yours), **Accept Incoming** (from GitHub) and **Accept Both**.
2. Keep the correct code and make sure all the `<<<<<<<`, `=======` and `>>>>>>>` lines are gone. If you are not sure which version is right, ask the person who wrote the other one.
3. Save the file, then:

```bash
git add path/to/the_file.py
git commit -m "Resolve conflict in the_file.py"
git push
```

## Common messages

| Git says… | Do this |
|---|---|
| `Your local changes ... would be overwritten by merge` (on `git pull`) | Commit your changes first, then pull. Or put them aside: `git stash`, `git pull`, `git stash pop`. |
| `rejected ... fetch first` (on `git push`) | `git pull`, then `git push` again. |
| `CONFLICT (content): Merge conflict in ...` | See "Fixing a conflict" above. |

## Cheat sheet

| I want to… | Command |
|---|---|
| See what I changed | `git status`, then `git diff` |
| See the history | `git log --oneline` |
| Undo `git add` (keep my changes) | `git restore --staged path/to/file` |
| Put my changes aside for a moment | `git stash`, and later `git stash pop` |
| Throw away my changes to a file (**can't be undone**) | `git restore path/to/file` |
