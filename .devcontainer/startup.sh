if [ -f requirements.txt ]; then
  pip install uv
  uv pip install --user -r requirements.txt
fi