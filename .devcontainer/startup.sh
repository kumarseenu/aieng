if [ -f requirements.txt ]; then
  pip install uv
  uv pip install --system -r requirements.txt
fi