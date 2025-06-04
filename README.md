# PassCompass 🧭 ✅
***MLOps Project | Spotting at-risk students before grades slip, guiding learners toward success.***

This project delivers an end-to-end, monitored pipeline (data ingest → training → deployment → drift alerts) that showcases modern **MLOps practices** and creates a tool that can be adapted by real schools with only spreadsheet-level infrastructure.

## Problem
Each semester, thousands of students silently accumulate risk factors, including poor attendance, high failure rates, and limited study time. These factors can culminate in course failure or dropping out. Most schools still rely on end-of-term grades to identify struggling students, when intervention is already too late.

## Goal
Build a lightweight, open-source prediction service that flags students with a high probability of failing a course early in the term, so teachers, counsellors, or mentoring programmes can intervene with targeted support (extra coaching, social-emotional resources, family outreach).



### Social relevance

| Angle                     | Why it matters                                                                                                                                                                |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Equity & inclusion**    | Low-income and first-generation learners are over-represented among repeaters. Early alerts help close attainment gaps.                                                       |
| **Drop-out prevention**   | Each prevented failure boosts persistence rates, reducing long-term economic costs for both students and institutions.                                                        |
| **Resource optimisation** | Schools can triage scarce tutoring budgets toward the highest-risk cohort instead of blanket remediation.                                                                     |
| **Data transparency**     | By using an academic, publicly licensed dataset and logging every experiment, the project demonstrates *explainable*, reproducible AI rather than opaque “black-box scoring.” |


## Flow diagram
```
┌────────────────────────┐
│   UCI Student CSVs     │
│  (Math & Portuguese)   │
└────────────┬───────────┘
             │  Prefect task: download
             ▼
┌────────────────────────┐
│  Data Cleaning & EDA   │
│  • join courses        │
│  • engineer target     │
└────────────┬───────────┘
             │  Prefect task: preprocess → save students_clean.csv
             ▼
┌────────────────────────┐
│  MLflow Experiment     │
│  Baseline Pipeline     │
│  • one-hot / scale     │
│  • logistic model      │
│  • 5-fold CV metrics   │
└────────────┬───────────┘
             │  registers best model in MLflow Registry
             ▼
┌────────────────────────┐
│  Docker Image Build    │
│  (GitHub Actions)      │
└────────────┬───────────┘
             │  push to GHCR
             ▼
┌───────────────────────────────┐
│  FastAPI Micro-service        │
│  • POST /predict ↦ pass prob  │
│  • loads model from registry  │
└────────────┬──────────────────┘
             │  JSON request/response
             ▼
       ┌──────────────┐
       │  End Users   │
       │  (teachers)  │
       └──────────────┘
             ▲
             │  logs inferences
┌────────────┴──────────────────┐
│      Evidently Monitoring     │
│  • drift & accuracy reports   │
│  • Prefect alert → Slack      │
└───────────────────────────────┘
```

Infra notes:
Terraform (not shown) provisions a small CPU VM + object storage bucket.

CI/CD: GitHub Actions lints, tests, builds the Docker image, and redeploys the FastAPI container on success.

Scalability: everything is stateless; swap the VM for Kubernetes later without code changes.
