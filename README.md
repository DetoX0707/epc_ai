# Data Centre EPC AI Intelligence Platform

 AI Intelligence Platform for Data Centre EPC Project Delivery.

## What This Project Demonstrates

This prototype implements your multi-agent MVP:

- Supervisor Agent: routes search, document ingestion, and audit workflows.
- Project Knowledge and RFI Agent: searches master specifications, RFIs, submittals, and project records using hybrid retrieval.
- Specification Compliance Auditor: extracts technical parameters from a vendor submittal, retrieves the relevant master requirements, and produces a compliance report with citations.

The app is intentionally local-first. It can run without paid APIs, Docker, or internet access. If you set `OPENAI_API_KEY`, the Polish Report button can turn the audit JSON into a sharper executive report, but the current demo is fully functional without it.

The current version is a Python Streamlit control-tower demo, not a plain chatbot. It includes:

- Command Center dashboard for equipment, risk, commissioning readiness, and evidence coverage.
- Ask tab for hybrid RFI/spec search with citations.
- Audit tab with three built-in vendor scenarios: cooling tower, UPS, and generator.
- Risk Board with dummy risk register and RFI impact ledger.
- Evidence explorer showing dense, sparse, and fused retrieval scores.
- Architecture tab mapping the Supervisor, Knowledge, and Compliance Auditor agents.

## Quick Start

From this folder:

```powershell
.\run.ps1
```

Then open:

```text
http://localhost:8501
```

If Streamlit is not installed:

```powershell
python -m pip install -r requirements.txt
```

## Demo Flow For Judges

1. Click "Load Demo Corpus".
2. Ask:

```text
What backup runtime is required for Tier III UPS systems under full load?
```

3. Open the Compliance Auditor tab.
4. Click "Run Demo Audit".
5. Show that the app finds deviations between the vendor cooling tower submittal and the project master specification in seconds.
6. Open the Architecture tab to explain why the workflow is scalable and business-relevant.


## Included Demo Documents

- `master_specification.txt`: master MEP requirements for UPS, cooling tower, generators, commissioning, and QA.
- `rfi_log.txt`: RFI decisions that reinforce UPS runtime, cooling tower acoustics, and BACnet requirements.
- `commissioning_checklist.txt`: IST acceptance criteria and required evidence.
- `change_order_014.txt`: acoustic requirement update for cooling tower cells.
- `procurement_tracker.txt`: critical equipment status and procurement gate.
- `vendor_cooling_tower_submittal.txt`: intentionally non-compliant vendor submission for the demo audit.
- `vendor_ups_submittal.txt`: non-compliant UPS autonomy and efficiency scenario.
- `vendor_generator_submittal.txt`: non-compliant generator fuel autonomy and event-log scenario.
- `equipment_register.csv`: dummy critical-equipment register for dashboard display.
- `risk_register.csv`: dummy EPC risk board.
- `commissioning_status.csv`: dummy integrated systems testing readiness data.
- `rfi_status.csv`: dummy RFI decision and rework-hours ledger.

## Project Structure

```text
streamlit_app.py         Python Streamlit app
app/
  epc_ai.py              Ingestion, hybrid retrieval, citations, audit logic
data/
  sample/
    master_specification.txt
    rfi_log.txt
    commissioning_checklist.txt
    change_order_014.txt
    procurement_tracker.txt
    vendor_cooling_tower_submittal.txt
  uploads/               Runtime uploads and generated index files
```


- `ARCHITECTURE.md`: Mermaid architecture diagram and judging-criteria mapping.
