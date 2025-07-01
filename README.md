# PassCompass 🧭 ✅ / DRAFT

## TO DO
- Revise makefile

## DECISIONS

- Check imbalance of classes


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

## The dataset
[UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student%2Bperformance)

Collected via questionnaires and school reports at two Portuguese secondary schools, this dataset captures socioeconomic, family and study-habit factors alongside three period grades (G1–G3) for 649 students in Mathematics and Portuguese language. It has become a staple benchmark for early-warning systems in education because it is small, tidy and publicly licensed, yet rich enough to test fairness and feature-drift monitoring strategies. They convert the final grade G3 into a binary pass (≥ 10) / fail target to align with real-world intervention workflows.


## Folder structure

PASSCOMPASS/
│
├── 00_exploration/              # existing notebooks
├── data/                        # raw + processed CSVs
│
├── **01_pipeline/**             # code that Prefect will call
│   ├── **train.py**             # train + log + register model
│   └── **prefect_flow.py**      # orchestration wrapper
│
├── **conf/**
│   └── **mlflow_local.yaml**    # central tracking config
│
├── **Makefile**                 # convenience targets
├── **environment.yml**          # include mlflow & prefect
├── **.env.example**             # env vars (ignored in .gitignore)
│
├── LICENSE
└── README.md

```bash
make env-create
make mlflow-ui          # open http://127.0.0.1:5000
make run-flow           # kicks off the Prefect pipeline
```


Schema:

| Raw column   | Suggested UI label               | Optional help / tooltip                       |             |             |             |             |
| ------------ | -------------------------------- | --------------------------------------------- | ----------- | ----------- | ----------- | ----------- |
| `school`     | **School**                       | GP = Gabriel Pereira · MS = Mousinho Silveira |             |             |             |             |
| `course`     | **Course**                       | Math or Portuguese                            |             |             |             |             |
| `sex`        | **Gender**                       | F = Female · M = Male                         |             |             |             |             |
| `age`        | **Age (years)**                  | Integer 15 – 22                               |             |             |             |             |
| `address`    | **Home address**                 | U = Urban · R = Rural                         |             |             |             |             |
| `famsize`    | **Family size**                  | LE3 = ≤ 3 · GT3 = > 3                         |             |             |             |             |
| `Pstatus`    | **Parents’ cohabitation**        | T = Together · A = Apart                      |             |             |             |             |
| `Medu`       | **Mother’s education level**     | 0 None                                        | 1 Primary   | 2 5-9 yrs   | 3 Secondary | 4 Higher Ed |
| `Fedu`       | **Father’s education level**     | Same scale as above                           |             |             |             |             |
| `Mjob`       | **Mother’s job**                 | Teacher, Health, Services, etc.               |             |             |             |             |
| `Fjob`       | **Father’s job**                 | —                                             |             |             |             |             |
| `reason`     | **Reason for choosing school**   | Close to home, Reputation, Course, Other      |             |             |             |             |
| `guardian`   | **Primary guardian**             | Mother, Father, Other                         |             |             |             |             |
| `traveltime` | **Daily travel time to school**  | 1 < 15 min                                    | 2 15-30 min | 3 30-60 min | 4 > 1 h     |             |
| `studytime`  | **Weekly study time**            | 1 < 2 h                                       | 2 2-5 h     | 3 5-10 h    | 4 > 10 h    |             |
| `failures`   | **Past class failures**          | 0 – 3 +                                       |             |             |             |             |
| `schoolsup`  | **Extra school support**         | Yes / No                                      |             |             |             |             |
| `famsup`     | **Family study support**         | Yes / No                                      |             |             |             |             |
| `paid`       | **Paid extra classes**           | Yes / No                                      |             |             |             |             |
| `activities` | **Extracurricular activities**   | Yes / No                                      |             |             |             |             |
| `nursery`    | **Attended nursery school**      | Yes / No                                      |             |             |             |             |
| `higher`     | **Wants higher education**       | Yes / No                                      |             |             |             |             |
| `internet`   | **Internet at home**             | Yes / No                                      |             |             |             |             |
| `romantic`   | **In a romantic relationship**   | Yes / No                                      |             |             |             |             |
| `famrel`     | **Family relationship quality**  | 1 Poor – 5 Excellent                          |             |             |             |             |
| `freetime`   | **Free time after school**       | 1 Very low – 5 Very high                      |             |             |             |             |
| `goout`      | **Going-out frequency**          | 1 Rarely – 5 Daily                            |             |             |             |             |
| `Dalc`       | **Week-day alcohol consumption** | 1 None – 5 Heavy                              |             |             |             |             |
| `Walc`       | **Weekend alcohol consumption**  | 1 None – 5 Heavy                              |             |             |             |             |
| `health`     | **Current health status**        | 1 Very bad – 5 Very good                      |             |             |             |             |
| `absences`   | **School absences (total)**      | Integer 0 – 93                                |             |             |             |             |
