# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "Setting up dependencies with uv..."

# Add uv to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# If requirements.txt exists, install dependencies with uv
if [ -f requirements.txt ]; then
  # Create a lock file if it doesn't exist or if requirements.txt is newer
  if [ ! -f requirements.lock ] || [ requirements.txt -nt requirements.lock ]; then
    echo "Generating lock file from requirements.txt..."
    uv pip compile requirements.txt -o requirements.lock
  fi

  # Create a virtual environment
  uv venv
  
  # Activate it
  . .venv/bin/activate
  
  # Install jupyter and ipykernel if they're not in requirements.txt
  grep -q "jupyter" requirements.txt || uv pip install jupyter
  grep -q "ipykernel" requirements.txt || uv pip install ipykernel

  echo "Installing dependencies from lock file..."

  # Install using the lock file (faster)
  uv pip install -r requirements.lock
fi

echo "Environment setup complete!"