# EduPro Student Segmentation and Course Recommendation System

This project builds a Streamlit analytics app for the `EduPro Online Platform.xlsx` dataset.
It creates learner-level profiles, segments students with clustering, validates the cluster
choice, and recommends personalized courses using segment-aware content scoring.

## Dataset

Keep `EduPro Online Platform.xlsx` in the project root. The app validates these required sheets:

- `Users`: `UserID`, `Age`, `Gender`
- `Courses`: `CourseID`, `CourseName`, `CourseCategory`, `CourseType`, `CourseLevel`, `CourseRating`
- `Transactions`: `UserID`, `CourseID`, `TransactionDate`, `Amount`

## Methodology

- Aggregates transactions into learner profiles.
- Engineers engagement, preference, spending, diversity, and learning-depth features.
- Scales numerical features and one-hot encodes categorical preferences.
- Selects K-Means cluster count from 3-7 segments with silhouette diagnostics and elbow context.
- Validates segmentation with hierarchical clustering silhouette.
- Scores recommendations using course rating, popularity within the learner segment,
  category match, level match, and course type match.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub, including `app.py`, `requirements.txt`, `runtime.txt`,
   `.streamlit/config.toml`, `README.md`, and `EduPro Online Platform.xlsx`.
2. In Streamlit Community Cloud, select the GitHub repository.
3. Set the main file path to `app.py`.
4. Deploy.

The app reads the Excel file with a path relative to `app.py`, so it works the same locally and
on Streamlit Cloud when the workbook is committed to the repository.

## Deliverables Covered

- Research-style EDA and insight summary inside the `Insights` tab.
- Live Streamlit dashboard with cluster visualization, profile explorer, segment comparison,
  and personalized recommendations.
- Executive stakeholder summary in the `Insights` tab.
