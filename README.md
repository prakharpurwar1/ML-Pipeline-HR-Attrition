# AttritionIQ — Employee Attrition Prediction & HR Analytics

**An end-to-end machine learning project for employee attrition prediction, model evaluation, and data-driven retention planning.**

AttritionIQ is an HR analytics application developed as part of the *Employee Attrition Prediction and Retention Strategy* Concurrent Running Project (CRP). It explores how machine learning can help organizations identify patterns associated with employee turnover and support proactive, evidence-based retention decisions.

The project combines a machine learning pipeline with a web-based dashboard for accessing employee attrition analytics and prediction functionality.

## Project Objectives

- Develop predictive models to identify employees who may be at risk of leaving an organization.
- Evaluate model performance using relevant classification metrics.
- Understand the factors associated with employee attrition.
- Translate analytical findings into actionable insights for HR teams and managers.
- Explore how AI-powered analytics can support employee engagement, workforce planning, and retention strategies.

## Key Features

- **Machine Learning Pipeline:** Organized workflows for data download, preprocessing, model training, evaluation, deployment, monitoring, and cleanup.
- **Attrition Prediction:** A web interface for submitting individual employee information for prediction.
- **Batch Prediction:** A separate interface for processing multiple employee records.
- **Feature Insights:** Access to information about the features used by the prediction system.
- **HR Analytics Dashboard:** Pages for exploring model analytics and monitoring-related information.
- **Model Evaluation:** Support for assessing predictive performance through classification metrics.
- **Cloud Deployment:** Project configuration for integrating a deployed AWS SageMaker inference endpoint.

Feature availability depends on the current model artifacts, backend configuration, and deployment environment.

## Machine Learning Approach

The academic project proposes comparing the following models:

| Model | Primary Purpose |
|---|---|
| Decision Tree | Interpretable prediction rules and HR insights |
| Random Forest | Ensemble prediction and feature importance analysis |
| XGBoost | Gradient-boosted prediction and performance benchmarking |
| Artificial Neural Network (ANN) | Exploring complex, nonlinear relationships |

The model comparison focuses on predictive performance, interpretability, complexity, scalability, and practical applicability to HR decision-making.

Evaluation metrics include accuracy, precision, recall, F1-score, and ROC-AUC, where applicable.

The final model selection should be based on measured evaluation results and business requirements rather than accuracy alone.

## Technology Stack

- **Programming:** Python
- **Machine Learning:** Scikit-learn and other model-specific libraries used by the pipeline
- **Backend:** Flask
- **Application Server:** Gunicorn
- **Frontend:** HTML, CSS, JavaScript
- **Containerization:** Docker and Docker Compose
- **Cloud Integration:** AWS SageMaker and related AWS services
- **Version Control:** Git and GitHub

## Repository Structure

```text
ML-Pipeline-HR-Attrition/
├── .github/
│   └── workflows/
├── frontend-app/
│   ├── backend/
│   │   ├── app.py
│   │   └── api.py
│   ├── frontend/
│   │   ├── static/
│   │   │   ├── css/
│   │   │   └── js/
│   │   └── templates/
│   │       ├── index.html
│   │       ├── batch.html
│   │       ├── analytics.html
│   │       └── monitor.html
│   ├── Dockerfile.api
│   ├── Dockerfile.web
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── README.md
├── src/
│   ├── data/
│   │   ├── data_download.py
│   │   └── data_preprocess.py
│   ├── model/
│   │   └── model_train.py
│   ├── evaluation/
│   │   └── model_evaluate.py
│   ├── deployment/
│   │   └── model_deploy.py
│   └── inference/
│       ├── inference.py
│       └── predict.py
├── .gitignore
├── HR Analytics Data.csv
├── README.md
└── requirements.txt
```

*Note: The structure above summarizes the principal project components. Individual files, workflow names, and generated model artifacts may vary with the current repository version.*

## Getting Started

### Prerequisites

- Git
- Python compatible with the project dependencies
- Docker and Docker Compose, if using the containerized setup
- AWS credentials and a deployed SageMaker endpoint, if using cloud inference

### 1. Clone the repository

```bash
git clone https://github.com/prakharpurwar1/ML-Pipeline-HR-Attrition.git
cd ML-Pipeline-HR-Attrition
```

### 2. Inspect the frontend configuration

```bash
cat frontend-app/README.md
cat frontend-app/docker-compose.yml
cat frontend-app/requirements.txt
```

Review the setup instructions and required environment variables before launching the application.

### 3. Configure environment variables

The frontend configuration uses environment variables such as:

```dotenv
AWS_REGION=ap-south-1
ENDPOINT_NAME=your-sagemaker-endpoint-name
```

Depending on the deployment configuration, AWS credentials and an S3 bucket may also be required.

Provide valid credentials through an appropriate secure mechanism. Do not commit AWS access keys, secret keys, session tokens, or other credentials to GitHub.

### 4. Run the web application

Follow the instructions in `frontend-app/README.md` to start the application.

For a Docker-based setup, the frontend's Compose configuration maps port `5000` to the application. Once the service is running, open the forwarded port in GitHub Codespaces.

The application may load without a configured inference endpoint, but prediction functionality requires a working backend and the appropriate model or cloud endpoint.

## Model Evaluation

Model evaluation is an important part of this project because employee attrition datasets can have imbalanced target classes.

The evaluation framework considers:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted attrition cases that are actual attrition cases.
- **Recall:** Proportion of actual attrition cases identified by the model.
- **F1-score:** Balance between precision and recall.
- **ROC-AUC:** Ability to distinguish between the two outcome classes across classification thresholds.

For HR retention planning, recall and precision should be considered alongside accuracy because missed attrition cases and unnecessary interventions have different business consequences.

## Business Value

AttritionIQ aims to connect predictive analytics with practical HR decision-making.

Potential applications include:

- Identifying workforce segments that may need closer attention.
- Understanding patterns associated with employee turnover.
- Supporting employee engagement and retention planning.
- Helping HR teams prioritize further investigation and appropriate interventions.
- Encouraging evidence-based workforce planning.

Predictions should be treated as decision-support signals, not definitive judgments about individual employees. Any HR intervention should consider additional context, fairness, privacy, and human review.

## Project Scope and Limitations

The project uses an HR analytics dataset to explore employee attrition prediction. Model performance and practical usefulness depend on data quality, feature selection, model validation, and deployment configuration.

Predicted attrition risk does not establish why an employee will leave, and statistical associations do not necessarily imply causation. The system should complement, rather than replace, professional HR judgment.

Cloud prediction functionality also depends on the availability and correct configuration of the required AWS resources.

## Future Enhancements

- Compare model performance using a consistent evaluation framework.
- Improve explainability through feature importance and prediction-level explanations.
- Expand dashboard visualizations for HR stakeholders.
- Strengthen monitoring and model performance tracking.
- Evaluate fairness and responsible use in employee-related predictions.
- Improve deployment reliability and automated testing.

## Academic Context

**Project:** Employee Attrition Prediction and Retention Strategy  
**Project Type:** Concurrent Running Project (CRP)  
**Domain:** Human Resource Analytics, Machine Learning, and Predictive Analytics

The project aims to bridge data science and business strategy by exploring how predictive models can inform proactive employee retention planning.

## License

No license has been specified for this repository. Add an appropriate license if you intend to distribute or permit reuse of the project.
