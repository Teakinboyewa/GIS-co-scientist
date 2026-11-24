
import sys
import os

project_root = '/home/<your-username>/<your-site>'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. Provide the OpenAI API key ---------------------------------
# Prefer setting this in the PythonAnywhere Web tab "Environment
# variables" section. As a fallback you can hardcode it here, but do
# NOT commit a real key to source control.
# os.environ['OPENAI_API_KEY'] = 'sk-...'

# --- 3. Expose the Flask app as `application` ----------------------
# PythonAnywhere's WSGI server looks for a callable named `application`.
from WebUI.app import app as application
