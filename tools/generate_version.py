import subprocess
import datetime
import re

def get_git_version():
    try:
        # Get branch name
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        
        # Get commit date (author date)
        commit_date = subprocess.check_output(
            ['git', 'show', '-s', '--format=%ai', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        
        # Get commit hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True
        ).strip()
        
        # Parse and format date
        dt = datetime.datetime.strptime(commit_date.split()[0], "%Y-%m-%d")
        date_str = dt.strftime("%Y.%m.%d")
        
        # Clean branch name
        branch_clean = re.sub(r'[^a-zA-Z0-9._-]', '.', branch)
        if branch in ['main', 'master']:
            branch_part = branch
        else:
            branch_part = branch_clean
        
        # Check if there are uncommitted changes
        dirty = subprocess.call(
            ['git', 'diff', '--quiet'],
            stderr=subprocess.DEVNULL
        ) != 0
        
        dirty_suffix = ".dirty" if dirty else ""
        
        return f"{date_str}+{branch_part}"
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        now = datetime.datetime.now()
        date_str = now.strftime("%Y.%m.%d")
        return f"{date_str}+nogit"

version = get_git_version()
print(f"Version: {version}")

# Write to version file
with open('omni_cli/_version.py', 'w') as f:
    f.write(f'__version__ = "{version}"\n')