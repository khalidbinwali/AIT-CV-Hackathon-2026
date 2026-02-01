# 🐙 How to Push to GitHub (Troubleshooting)

It seems `git` is not recognized mainly because it's not in your system "Path". This is common on Windows.

## Option 1: Use "Git Bash" (Recommended)
If you have Git installed, you likely have a program called **Git Bash**.

1.  Press `Windows Key`, type **"Git Bash"**, and open it.
2.  Navigate to your project folder:
    ```bash
    cd "c:/Users/ZAH/Desktop/Hackathon_Dataset/Hackathon2_scripts"
    ```
    *(Note the forward slashes `/` used in Git Bash)*
3.  Run the commands to push:
    ```bash
    git init
    git add .
    git commit -m "Hackathon Submission"
    git branch -M main
    git remote add origin https://github.com/khalidbinwali/AIT-CV-Hackathon-2026.git
    git push -u origin main
    ```

## Option 2: Install Git (If validation fails)
If you don't find "Git Bash", you need to install standard Git:
1.  Download from [git-scm.com/download/win](https://git-scm.com/download/win).
2.  Run the installer. **Important:** When asked about "Adjusting your PATH environment", select **"Git from the command line and also from 3rd-party software"**.
3.  After install, restart your terminal (PowerShell or VS Code) and try again.

## Option 3: GitHub Desktop (Visual Interface)
1.  Download [GitHub Desktop](https://desktop.github.com/).
2.  Open it and sign in.
3.  Go to **File** > **Add Local Repository**.
4.  Select this folder: `c:\Users\ZAH\Desktop\Hackathon_Dataset\Hackathon2_scripts`
5.  Click **Publish repository** to push it online.

---
**Need to check if your files are ready?**
Your project is fully prepped with:
- `.gitignore` (Already created)
- `HACKATHON_REPORT.md` (Already created)
- All script files
