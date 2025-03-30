# if [ -f requirements.txt ]; then
#   pip install --upgrade pip
#   pip install uv
#   pip install --system -r requirements.txt
# fi

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (if needed)
export PATH="$HOME/.cargo/bin:$PATH"

# If requirements.txt exists, install dependencies with uv
if [ -f requirements.txt ]; then
  # Create a virtual environment
  uv venv
  # Activate it and install requirements
  . .venv/bin/activate
  
  # Install jupyter and ipykernel if they're not in requirements.txt
  uv pip install jupyter ipykernel

  uv pip install -r requirements.txt
fi