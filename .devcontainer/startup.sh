if [ -f requirements.txt ]; then
  pip install uv
  pip install --system -r requirements.txt
fi