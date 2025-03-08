# Template Kit for RAMP challenge

[![Build status](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml/badge.svg)](https://github.com/ramp-kits/template-kit/actions/workflows/test.yml)

## Introduction

At the start of the year, the **Court of Auditors** published its report on the evaluation of the **Skills Investment Plan**, implemented during Emmanuel Macron’s first term from 2017 to 2022. This plan allocated **15 billion euros** over five years to tackle youth and long-term unemployment. However, the report highlights an **uneven distribution of funds** across regions and years, with some areas receiving surplus funding while others faced shortages.  

### 🎯 Project Objective  

Our project proposes **a solution for reallocating resources annually based on predicted funding needs at the departmental level**. At the end of each year **N-1**, we aim to **forecast the number of young job seekers for year N**.  

This prediction will help the **State better assess regional training funding needs**, as they must depend on the number of unemployed individuals. By doing so, we can **optimize financial resource allocation between regions** and improve the **efficiency of future public policies**.  

### Scope & Target Variable  

For this project, we focus on **young job seekers at the departmental level**. Our **target variable** is:  

> **"The number of job seekers under 25 years old in department D for year N".**  

To build this prediction, we leverage the **Workforce Needs Survey** conducted by **France Travail** in the last quarter of each year. This survey gathers data from **2 million private-sector companies**, asking about:  

- **Expected job creations** for the upcoming year  
- **Challenges in filling positions** (e.g., skill shortages, job difficulty)  

Additionally, we integrate other **key indicators** from year **N-1**, including:  

- **Number of job postings**  
- **Completed training programs**  
- **Control variables** such as **departmental population**  

This data-driven approach will enable a **more equitable distribution of funding** and contribute to a **more efficient labor market policy**.  

## 📊 Dataset Description  

This dataset is designed to **predict youth unemployment (ages 15-24) in France** using indicators from the **French labor market**. The data comes from **France Travail**, specifically from the **Statistiques et Analyses** section, covering the period **2015 to 2023**.  

### 🏷️ Key Features  

The dataset includes the following features:  

- **Year**  
- **Department**  
- **Workforce needs declared by companies** → Indicates **recruitment demand** across different sectors for this year and department.  
- **Recruitment difficulty index (0-100%)** → Shows the **percentage of difficulty companies face** when hiring for this year and by department.  
- **Number of unemployed youth (15-24 years old)** → Recorded for the **previous year** and by department.  
- **Number of training programs offered** → Indicates **workforce skill development** for job seekers in the **previous year** and by department.  
- **Number of job offers available** → Reflects **labor market demand** for the **previous year** and by department.  
- **Number of people entering and exiting unemployment** → Captures **job market inflows and outflows** for the **previous year** and by department.  
- **Population size** → Data from *INSEE*, recorded for the **previous year** and by department.  

This dataset provides a **comprehensive view of labor market dynamics**, helping to forecast youth unemployment and optimize resource allocation.  


## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started on this RAMP with the
[dedicated notebook](template_starting_kit.ipynb).

### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
