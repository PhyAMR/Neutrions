# Local Git Workflow Guide

Because this repository is being built by multiple developers sharing the **same physical machine** (and currently without a remote like GitHub), it is vital to coordinate work carefully to avoid overwriting code.

## 1. Environment & Workspace
*If you are all using the same user account and the exact same folder (e.g., `/home/ubuntu/Grassi`):*
- **Communication is key**: You must coordinate visually or verbally to ensure you aren't actively typing in the exact same file at the same time.
- **VS Code Live Share**: If you are connecting from different machines via SSH, consider heavily utilizing VS Code Live Share to edit the same workspace collaboratively in real-time.

*Ideally*, you should initiate a "bare" repo on the machine (`git init --bare /home/ubuntu/Grassi.git`) and each user can `git clone` that into their own separate directories. However, Assuming you are using one shared folder, follow the branching strategy below.

## 2. Branching Strategy
**Never work directly on the `main` branch.**
When you start a new task, create a new branch using a prefix with your initials or the feature type:
```bash
# Ensure you are up to date and on main
git checkout main

# Create and switch to your new branch
git checkout -b feature/ag-add-preprocessing
```

## 3. Making Commits
Save your progress into Git incrementally. Only commit code that runs!
```bash
# See what files you've modified
git status

# Stage the specific files you want to commit
git add src/data_loader.py

# Commit with a clear, descriptive message
git commit -m "Add handling for missing values in data loader"
```

## 4. Merging Your Work
Once your script or feature is working perfectly, it's time to integrate it back to `main`.

1. **Switch to main**:
   ```bash
   git checkout main
   ```
2. **Merge your feature branch**:
   ```bash
   git merge feature/ag-add-preprocessing
   ```
3. **Delete your branch** (keep the workspace clean!):
   ```bash
   git branch -d feature/ag-add-preprocessing
   ```

## 5. Dealing with Changes and Conflicts
If two people modified the *same file*, Git will trigger a **Merge Conflict**. 

1. When you run `git merge`, Git will warn you about the conflict.
2. Open the conflicted file in VS Code.
3. You will see markers separating the two versions of the code:
   ```python
   <<<<<<< HEAD
   logger.info("This is the code currently on main")
   =======
   logger.info("This is the code from your feature branch")
   >>>>>>> feature/ag-add-preprocessing
   ```
4. **Discuss with your teammate** on which version is correct, or manually combine them into the right logic.
5. In VS Code, you can click the convenient "Accept Current Change", "Accept Incoming Change", or "Accept Both Changes" buttons above the conflict.
6. Once the file looks exactly how it should, save it.
7. Stage and complete the merge:
   ```bash
   git add <resolved_file.py>
   git commit -m "Resolve merge conflict in <file>"
   ```

## 6. Pro-Tips for Shared Machines
- **`git stash`**: If you need to switch branches but aren't ready to commit half-broken code, run `git stash`. You can retrieve it later with `git stash pop`.
- **VS Code Source Control Tab**: Use the built-in UI on the left sidebar for visually reviewing changes, staging, committing, and handling merge conflicts safely.
