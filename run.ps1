$ErrorActionPreference = "Stop"

$python = "C:\Users\PRATIK\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (!(Test-Path $python)) {
  $python = "python"
}

$hasStreamlit = & $python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('streamlit') else 1)"
if ($LASTEXITCODE -ne 0) {
  Write-Host "Streamlit is not installed in this Python runtime."
  Write-Host "Install it with: $python -m pip install -r requirements.txt"
  exit 1
}

& $python -m streamlit run ".\streamlit_app.py" --server.port 8501 --server.address localhost --server.headless true --browser.gatherUsageStats false
