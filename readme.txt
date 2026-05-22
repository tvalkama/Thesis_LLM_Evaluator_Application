Windows setups:
##cd to project folder
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## run
python task_creator_gradio.py

## Convert confluence mhtml to txt (and strip images etc special formatting)
confluence_to_plaintext.py "SAP+AWS+Server+Infrastructure+Upgrade.doc" sap_aws_server_infrastructure_upgrade.txt