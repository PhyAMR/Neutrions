# Git workflow

The code lives on GitHub at `git@github.com:PhyAMR/Neutrions.git`. **Nobody pushes to `main` directly.** Every change goes through a *pull request* (PR):

1. you work on your own branch;
2. you push it and open a PR;
3. a teammate reads it, tries it and approves it;
4. it is merged into `main`.

Each of you works on your own VM, in your own copy of the repository (a *clone*). **GitHub is the only place where work is shared**: if you haven't pushed it, the others can't see it.

Why PRs:
- `main` keeps running, because a second person has read and run every change before it lands.
- If two people change the same file, they see it in the PR, before it breaks `main`.
- The discussion stays on the PR page, next to the code, so the reason for a change isn't lost.

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
3. Ask the owner of the repository (`PhyAMR` on GitHub) to add you as a collaborator, and accept the invitation that GitHub emails you. Collaborators can push branches to the shared repository, so you don't need a fork.
4. Test it: `ssh -T git@github.com` should greet you by your GitHub username.

### 3. Clone the repository

```bash
cd ~
git clone git@github.com:PhyAMR/Neutrions.git Grassi
cd Grassi
```

If you already cloned your own **fork**, point your clone at the shared repository instead (once you are a collaborator):

```bash
git remote set-url origin git@github.com:PhyAMR/Neutrions.git
git fetch origin
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

### 5. Protect `main` (owner only, once)

On GitHub: **Settings** → **Branches** → **Add classic branch protection rule**. Set the branch name pattern to `main`, tick **Require a pull request before merging** (with **Require approvals: 1**) and **Do not allow bypassing the above settings**, then save. From then on, GitHub rejects a direct `git push` to `main`, even from the owner. (Done for this repository on 2026‑11‑09. Force-pushes to `main` and deleting it are blocked too.)

## Every time you work

```bash
# 1. Start from an up-to-date main
git switch main
git pull

# 2. Make a branch for this piece of work: one topic per branch
git switch -c yourname/short-topic       # e.g. alvaro/config-reader

# 3. Work, and save snapshots as you go
git status                     # what did I change?
git add src/some_file.py       # choose the files to save
git commit -m "Say in one line what changed"

# 4. Upload the branch (-u only the first time; after that, just `git push`)
git push -u origin yourname/short-topic
```

**Open the PR.** After the push, git prints a link (`Create a pull request for 'yourname/short-topic' on GitHub by visiting: …`). Open it, or click **Compare & pull request** on the GitHub page. **Check the two branch boxes at the top: `base` must be `main`, and `compare` must be your branch.** A PR into any other branch never reaches `main`. In the description, say what changed and how you checked it (what you ran and what came out). Then pick a reviewer on the right.

While the PR is open, keep committing and pushing to the same branch. The PR updates by itself.

**After the PR is merged:**

```bash
git switch main
git pull
git branch -d yourname/short-topic
```

Start your next piece of work with a new branch from here. Don't keep working on a branch that has already been merged.

## Reviewing a pull request

1. On the PR page, open **Files changed** and read the diff. Click the **+** next to a line to comment on it.
2. Run it on your VM before you approve. Commit or stash your own changes first, then:

   ```bash
   git fetch origin
   git switch theirname/short-topic    # the branch name is at the top of the PR page
   uv run main.py                      # or whatever the PR says it changes
   git switch -                        # back to the branch you were on
   ```

   If the PR comes from a fork, fetch it by number instead (here PR 7): `git fetch origin pull/7/head:pr-7`, then `git switch pr-7`.
3. Click **Review changes**, then either **Approve**, or **Request changes** and say what to fix.
4. Once it is approved, click **Merge pull request** (keep the default, "Create a merge commit"), then **Delete branch**.

## Keeping your branch up to date

If `main` moved while your PR was open, GitHub says "This branch is out-of-date with the base branch" or "This branch has conflicts". To fix either one, merge `main` into your branch:

```bash
git switch yourname/short-topic
git pull origin main
git push
```

If this reports a conflict, fix it as described below, then push.

## Rules

- **Never commit on `main`.** Start every piece of work with `git switch -c`. Check where you are with `git branch --show-current`.
- **One topic per PR, and keep it small.** A teammate can review a short PR the same day; a 1,000‑line PR waits.
- **Someone else approves before you merge.**
- **Say in the PR how you checked it.**
- **Add only the files you meant to change.** Avoid `git add .`, because it also picks up data, outputs and temporary files.
- **Clear notebook outputs before you commit a notebook** (in Jupyter or VS Code: *Clear All Outputs*), unless the output is the point of the PR. Embedded plots make the diff unreadable in a review.
- **Say when you're working on a shared file** (for example `src/scripts/preprocessing.py`), so two open PRs don't change the same lines.

## Fixing a conflict

When you and someone else changed the same lines, `git pull origin main` can't choose between them and marks the file:

```text
<<<<<<< HEAD
your version
=======
the version from main
>>>>>>> ...
```

1. Open the file. VS Code shows buttons above each conflict: **Accept Current** (yours), **Accept Incoming** (from `main`) and **Accept Both**.
2. Keep the correct code and make sure all the `<<<<<<<`, `=======` and `>>>>>>>` lines are gone. If you are not sure which version is right, ask the person who wrote the other one.
3. Save the file, then:

```bash
git add path/to/the_file.py
git commit -m "Merge main and resolve conflict in the_file.py"
git push
```

## Common messages

| Git says… | Do this |
|---|---|
| `Your local changes ... would be overwritten` (on `git pull` or `git switch`) | Commit your changes first, or put them aside: `git stash`, do the pull or switch, then `git stash pop`. |
| `protected branch hook declined` / `GH006` (on `git push`) | You committed on `main`. Move the commits to a branch: `git switch -c yourname/topic`, then `git branch -f main origin/main` (this resets `main` without touching your files), then `git push -u origin yourname/topic`. |
| `The current branch ... has no upstream branch` (on `git push`) | It's the first push of a new branch: `git push -u origin yourname/topic`. |
| `rejected ... fetch first` (on `git push`) | Someone else pushed to the same branch: `git pull`, then `git push` again. |
| `CONFLICT (content): Merge conflict in ...` | See "Fixing a conflict" above. |
| `The branch ... is not fully merged` (on `git branch -d`) | The PR isn't merged yet. Check the PR page. |

## Cheat sheet

| I want to… | Command |
|---|---|
| See which branch I'm on | `git branch --show-current` |
| Start a new piece of work | `git switch main`, `git pull`, `git switch -c yourname/topic` |
| See what I changed | `git status`, then `git diff` |
| See the history | `git log --oneline --graph` |
| Bring the latest `main` into my branch | `git pull origin main` |
| Undo `git add` (keep my changes) | `git restore --staged path/to/file` |
| Put my changes aside for a moment | `git stash`, and later `git stash pop` |
| Throw away my changes to a file (**can't be undone**) | `git restore path/to/file` |
| Delete a branch after its PR is merged | `git branch -d yourname/topic` |
