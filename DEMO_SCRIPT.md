# 3-Minute Hackathon Demo Script

## 0:00-0:30 - Set The Problem

Data centre EPC projects fail when information is fragmented. A vendor submittal may look acceptable in isolation, but the actual requirement is spread across the master specification, RFI responses, change orders, commissioning checklists, and procurement trackers.

Our prototype creates a project intelligence layer that searches those documents and automates first-pass specification QA.

## 0:30-1:15 - Show The Knowledge Agent

Open the Ask tab and ask:

```text
What backup runtime is required for Tier III UPS systems under full load?
```

Point out that the answer is not generic. It cites the project master specification and RFI evidence.

## 1:15-2:15 - Show The Compliance Auditor

Open the Audit tab and click Run Demo Audit.

Explain:

The uploaded vendor cooling tower submittal is automatically parsed. The auditor retrieves the matching cooling tower requirements from the project corpus, compares engineering values, and produces a quality report.

Highlight the findings:

- Ambient temperature submitted as 42 C versus required 46 C minimum.
- Leaving water temperature submitted as 33 C versus required 32 C maximum.
- Approach temperature submitted as 6 C versus required 5 C maximum.
- Fan motor power submitted as 52 kW versus required 45 kW maximum.
- Noise level submitted as 82 dB versus required 78 dB maximum.

## 2:15-2:45 - Show The Business Impact

Use this line:

"Traditionally, this review requires an engineer to open multiple PDFs and manually cross-check 100+ line items. Our workflow turns a 3-4 hour manual verification task into a cited automated first-pass audit in under a minute."

## 2:45-3:00 - Close With Scale

This MVP starts with cooling tower and UPS evidence, but the architecture extends to generators, switchgear, commissioning records, RFI history, change orders, and schedule risk. It is not a chatbot; it is an auditable EPC project intelligence layer.
