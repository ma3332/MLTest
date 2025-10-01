# Install Python 3.13.7

pyenv install 3.13.7

# Set up for global python version (optional)

pyenv global 3.13.7

# Remove Python version

pyenv uninstall 3.9.18

#####################################################################

# Create project directory

mkdir my_awesome_project
cd my_awesome_project

# Set Python version for this project specifically

pyenv local 3.13.7

# This creates a .python-version file

cat .python-version

# Shows: 3.13.7

# Create virtual environment

python -m venv venv

# Activate virtual environment

source venv/bin/activate

# Deactivate virtual environment

deactivate

# Install packages (example)

pip install --upgrade pip
pip install requests pandas numpy

# Create/ Update requirements file

pip freeze > requirements.txt

#####################################################################

# Daily Workflow

cd my_awesome_project
source venv/bin/activate # Activate virtual environment
pip install sth # optional
pip freeze > requirements.txt
