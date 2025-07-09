export interface Project {
    slug: string;
    title: string;
    description: string;
    tech_stack: string[];
    created_date: string;
    featured?: boolean;
    github_url?: string;
    demo_url?: string | null;
    image_name?: string;
    detailed_description?: string;
    challenges?: string;
    results?: string;
    // Add other fields as needed
}

export const projects: Project[] = [
{
    slug: 'semantic-book-recommender-with-llms',
    title: 'Semantic Book Recommender with LLMs',
    description: `Built a semantic search-powered book recommendation system using Large Language Models. Implemented vector search with ChromaDB to find books similar to natural language queries, performed zero-shot classification to label books as 'fiction' or 'non-fiction', and conducted sentiment and emotion analysis to assess tone (e.g., suspenseful, joyful, sad). Developed an interactive Gradio interface for real-time recommendations. Project includes robust preprocessing, vector embeddings, and OpenAI-powered language understanding.`,
    detailed_description: `Built a semantic search-powered book recommendation system using Large Language Models. Implemented vector search with ChromaDB to find books similar to natural language queries, performed zero-shot classification to label books as 'fiction' or 'non-fiction', and conducted sentiment and emotion analysis to assess tone (e.g., suspenseful, joyful, sad). Developed an interactive Gradio interface for real-time recommendations. Project includes robust preprocessing, vector embeddings, and OpenAI-powered language understanding.`,
    tech_stack: ['Python', 'LangChain', 'OpenAI', 'Transformers', 'ChromaDB', 'Gradio', 'Pandas', 'Seaborn'],
    github_url: 'https://github.com/Jafri115/llm-book-recommender',
    demo_url: 'https://huggingface.co/spaces/Wasifjafri/semantic-book-recommender',
    image_name: 'semantic-book-recommender-with-llms.png',
    created_date: '2025-06-23',
    featured: true,
    challenges: '',
    results: ''
},
{
    slug: 'stylesense-realtime-fashion-classifier',
    title: 'StyleSense – Real‑Time Fashion Classifier',
    description: `A Flask web application that performs real-time fashion image classification using TensorFlow InceptionV3 and Hugging Face ResNet-50 models. Includes Docker containers for scalable deployment and a RESTful API for batch inference.`,
    detailed_description: `*Personal Project*

### Project Overview

**Fashion Image Classification Web App** is a Flask-based web application that classifies grayscale fashion images from the Fashion MNIST dataset. The system processes uploaded images through custom deep learning pipelines and returns class predictions with confidence scores via an intuitive web interface.

### Key Innovation

**Fashion Image Classification** combines:
- **Dual Model Architecture**: Implementation of both **InceptionV3** (TensorFlow) and **ResNet-50** (Hugging Face) with comparative evaluation
- **Custom Feature Extractors**: Purpose-built wrapper layers for grayscale-to-RGB conversion and automatic resizing
- **End-to-End Pipeline**: Complete workflow from image upload to classification results with preprocessing visualization
- **Flexible Input Processing**: Supports multiple image formats (.png, .jpg, .jpeg) with automatic preprocessing

### Technical Architecture

#### Core Components
- **InceptionV3FeatureExtractor**: Scales 28×28 grayscale images to 299×299 RGB using Inception's convolutional base
- **ResNetFeatureExtractor**: Wraps Hugging Face ResNet backbone processing inputs in [batch, 3, 224, 224] format
- **Flask Web Service**: Handles file uploads, image processing, and JSON API responses

#### Technology Stack
- **Backend**: Flask, TensorFlow (Keras API), Hugging Face Transformers
- **Models**: InceptionV3 (pretrained on ImageNet), ResNet-50 (microsoft/resnet-50)
- **Image Processing**: Pillow, NumPy
- **Frontend**: HTML, CSS

### Performance Results

After comparative training and evaluation, **InceptionV3 yielded higher accuracy and generalization** on Fashion MNIST grayscale images compared to Hugging Face ResNet-50, leading to its selection as the primary model.

### Implementation Highlights

- **Custom Architecture Design**: Developed specialized wrapper layers to handle grayscale-to-RGB conversion and multi-scale image preprocessing
- **Model Comparison Framework**: Implemented and evaluated both TensorFlow and Hugging Face model ecosystems for optimal performance
- **Production Web Application**: Built complete Flask application with file upload handling, image preprocessing pipeline, and JSON response formatting
- **User Experience Features**: Integrated visualization of both original and preprocessed images alongside classification results and confidence scoring

*Demonstrates practical machine learning implementation with modern deep learning frameworks and full-stack web development capabilities.*`,
    tech_stack: ['Python', 'Flask', 'TensorFlow', 'Hugging Face', 'Docker'],
    github_url: 'https://github.com/Jafri115/huggingface-resnet-fmnist',
    demo_url: null,
    image_name: 'stylesense-real-time-fashion-classifier.png',
    created_date: '2025-06-23',
    featured: true,
    challenges: '',
    results: ''
},
{
    slug: 'caliprice-mlpowered-housing-forecast',
    title: 'CaliPrice – ML‑Powered Housing Forecast',
    description: `End-to-end machine learning pipeline for California housing price forecasting using Random Forest, Gradient Boosting, and Linear Regression models. Deployed via a Flask app on Heroku with Dockerized services and CI/CD configured through GitHub Actions.`,
    detailed_description: `*Full-Stack Machine Learning Application*

### Project Overview
CaliPrice is an end-to-end machine learning solution for predicting California house prices. The project demonstrates the complete ML lifecycle from data preprocessing to production deployment, featuring a web interface and automated CI/CD pipeline.

### Key Features
- **Multi-Model Training**: Implemented and compared Linear Regression, Random Forest, and Gradient Boosting algorithms
- **Flask Web Application**: Interactive interface for real-time price predictions with form-based input
- **Production Deployment**: Docker containerization with live Heroku deployment
- **CI/CD Pipeline**: Automated GitHub Actions workflow for seamless deployments

### Technical Stack
- **Backend**: Python, Flask, Scikit-learn
- **Data Processing**: Pandas, NumPy, feature engineering
- **Deployment**: Docker, Heroku, GitHub Actions
- **Model Evaluation**: R², MSE, and MAE metrics

### Performance Results
The optimized ensemble model demonstrates strong predictive performance on the California Housing dataset with comprehensive cross-validation ensuring generalization to new data.

### Implementation Highlights
- **End-to-End Pipeline**: Complete ML workflow from data preprocessing to model deployment
- **Production Ready**: Robust error handling, input validation, and containerized deployment
- **Automated Workflow**: GitHub Actions CI/CD with Heroku Container Registry integration
- **Clean Architecture**: Modular design separating data processing, modeling, and web layers

---
*This project showcases building production-ready ML applications with modern DevOps practices.*`,
    tech_stack: ['Python', 'Scikit-learn', 'Flask', 'Docker', 'Heroku', 'GitHub Actions'],
    github_url: 'https://github.com/Jafri115/end-to-end-ml-california-pricing',
    demo_url: null,
    image_name: 'caliprice-ml-powered-housing-forecast.png',
    created_date: '2025-06-23',
    featured: true,
    challenges: '',
    results: ''
},
{
    slug: 'motionlstm-deep-learning-synthesis-of-human-movement',
    title: 'MotionLSTM – Deep Learning Synthesis of Human Movement',
    description: `Research project using LSTM Autoencoders and Seq2Seq architectures to synthesize human motion trajectories. Data preprocessing includes noise augmentation and normalization, with hyperparameter tuning for optimal sequence prediction.`,
    detailed_description: `*Computer Vision Research Project - Human Motion Analysis*

### Project Overview

**Human Motion Synthesis using Attention-based Autoencoder** is a cutting-edge computer vision research project that generates and forecasts realistic human motion sequences. The solution combines attention mechanisms with autoencoder architectures to model complex temporal dependencies in 3D motion capture data, addressing limitations of traditional RNN-based approaches for long-term motion prediction.

### Key Innovation

**This research project** advances the state-of-the-art in human motion synthesis:
- **Attention-Enhanced Architecture**: Novel integration of scaled dot-product attention between encoder-decoder layers for improved long-range temporal dependency modeling
- **Hybrid Temporal Processing**: Combination of GRU-based sequence encoding with transformer attention mechanisms for optimal motion pattern capture
- **Multi-Modal Evaluation**: Comprehensive assessment using MPJPE and NPSS metrics on Human3.6M dataset with baseline comparisons
- **Autoregressive Prediction**: Advanced decoder architecture enabling future motion frame generation with temporal consistency

### Technical Architecture

The system implements a sophisticated attention-based neural architecture for motion synthesis:
- **Encoder Layer**: Stacked GRU networks for temporal sequence compression into latent representations
- **Attention Mechanism**: Scaled dot-product attention for capturing long-range dependencies across motion sequences
- **Decoder Layer**: Autoregressive prediction network generating future 3D joint positions
- **Loss Framework**: MSE optimization with optional bone-length regularization and NPSS constraints

**Technology Stack:**
- **Deep Learning**: PyTorch with custom attention layers and GRU/Transformer hybrid architectures
- **Data Processing**: Human3.6M dataset preprocessing with 3D joint position normalization and sequence windowing
- **Model Training**: Advanced optimization with configurable hyperparameters and checkpoint management
- **Evaluation**: Comprehensive metrics including MPJPE for spatial accuracy and NPSS for temporal consistency
- **Research Tools**: Jupyter notebooks for analysis, YAML configuration management, and automated experiment tracking

### Performance & Results

The model demonstrates superior performance in human motion synthesis benchmarks:
- **Temporal Modeling**: Attention mechanism significantly improves long-term dependency capture compared to traditional RNN approaches
- **Spatial Accuracy**: Competitive MPJPE scores on Human3.6M dataset with improved motion naturalness
- **Architecture Comparison**: Systematic evaluation against GRU, LSTM, and pure transformer baselines demonstrating hybrid approach advantages

### Implementation Highlights

- **Research-Grade Architecture**: Novel attention-autoencoder combination with theoretical foundations and empirical validation on standard benchmarks
- **Scalable Training Pipeline**: Configurable training framework with YAML-based hyperparameter management and automated checkpoint saving
- **Comprehensive Evaluation**: Multi-metric assessment framework including spatial accuracy (MPJPE) and temporal consistency (NPSS) measurements
- **Modular Design**: Flexible architecture supporting multiple encoder/decoder combinations with pluggable attention mechanisms for research extensibility

*This project demonstrates advanced computer vision research capabilities, combining novel neural architecture design with rigorous experimental validation for human motion understanding and synthesis.*`,
    tech_stack: ['Python', 'TensorFlow', 'Keras', 'LSTM', 'Seq2Seq'],
    github_url: 'https://github.com/Jafri115/human-motion-synthesis',
    demo_url: null,
    image_name: 'motionlstm-deep-learning-synthesis-of-human-movement.png',
    created_date: '2025-06-23',
    featured: false,
    challenges: '',
    results: ''
},
{
    slug: 'tpch-elt-orchestrator-snowflake-dbt-airflow',
    title: 'TPCH ELT Orchestrator – Snowflake, dbt & Airflow',
    description: `A scalable ELT pipeline on the TPCH dataset using Snowflake for data warehousing, dbt for transformation models, and Apache Airflow (via Astronomer Cosmos) for workflow orchestration and automated testing.`,
    detailed_description: `*Modern Data Engineering ELT Pipeline*

### Project Overview

**TPCH ELT Orchestrator** is a production-ready data engineering solution that demonstrates modern ELT (Extract, Load, Transform) patterns using industry-standard tools. The project transforms Snowflake's TPCH dataset through SQL-based transformations orchestrated by Apache Airflow, implementing dimensional modeling with comprehensive data quality testing and workflow automation.

### Key Innovation

**This ELT pipeline** showcases modern data engineering best practices:
- **SQL-First Transformation**: dbt Core for version-controlled, testable, and modular SQL transformations with built-in documentation generation
- **Local Development Environment**: Dockerized Airflow using Astronomer CLI enabling reproducible development and deployment workflows
- **Star Schema Implementation**: Dimensional modeling with fact and dimension tables optimized for analytical workloads
- **Automated Data Quality**: Comprehensive dbt testing framework with uniqueness, referential integrity, and accepted values validation

### Technical Architecture

The system implements a cloud-native ELT pattern with local orchestration capabilities:
- **Data Warehouse**: Snowflake with TPCH sample dataset for enterprise-scale data processing
- **Transformation Layer**: dbt Core with staging, intermediate, and mart model layers
- **Orchestration Layer**: Apache Airflow with custom DAG definitions and Snowflake connectivity
- **Development Environment**: Astronomer CLI with Docker containerization for consistent deployment

**Technology Stack:**
- **Data Warehouse**: Snowflake with role-based access control and warehouse compute scaling
- **Transformation**: dbt Core with Jinja templating, macros, and automated dependency resolution
- **Orchestration**: Apache Airflow with custom operators, connection management, and scheduling capabilities
- **Development**: Astronomer CLI, Docker containerization, and local development server
- **Data Quality**: dbt testing framework with schema validation and data profiling capabilities

### Performance & Results

The pipeline demonstrates robust data processing with enterprise-grade quality controls:
- **Transformation Efficiency**: Modular dbt models with staging-to-marts progression enabling incremental processing and selective rebuilds
- **Data Quality Assurance**: Comprehensive testing suite including unique constraints, not-null checks, and accepted values validation
- **Orchestration Reliability**: Airflow DAG execution with automated retry logic, dependency management, and monitoring capabilities

### Implementation Highlights

- **Modular Architecture**: Separation of staging, intermediate, and mart layers with clear data lineage and dependency management through dbt's DAG compilation
- **Version-Controlled Transformations**: Git-based dbt project structure with automated documentation generation and data catalog maintenance
- **Production-Ready Orchestration**: Dockerized Airflow deployment with Snowflake connection management and environment variable configuration
- **Comprehensive Testing**: Built-in data quality framework with schema validation, referential integrity checks, and automated test execution in CI/CD pipelines

*This project demonstrates modern data engineering practices, combining SQL-first transformation approaches with robust orchestration and quality assurance for scalable analytical data processing.*`,
    tech_stack: ['Snowflake', 'dbt', 'Apache Airflow', 'Astronomer Cosmos'],
    github_url: 'https://github.com/Jafri115/snowflake-dbt-airflow-elt-pipeline',
    demo_url: null,
    image_name: 'tpch-elt-orchestrator-snowflake-dbt-airflow.png',
    created_date: '2025-06-23',
    featured: false,
    challenges: '',
    results: ''
},
{
    slug: 'azureolympics-databricks-synapse-analytics-etl',
    title: 'AzureOlympics – Databricks & Synapse Analytics ETL',
    description: `End-to-end Azure data engineering project ingesting Olympic CSV datasets into Data Lake Gen2, transforming them in Databricks, and analyzing with Synapse Analytics for performance insights and dashboards.`,
    detailed_description: `

### Project Overview

**AzureOlympics** is a comprehensive Azure-based data engineering solution that transforms raw Olympic datasets into interactive business intelligence insights. The project demonstrates modern cloud data architecture, processing Tokyo 2021 Olympic data through automated ETL pipelines to deliver interactive dashboards analyzing athlete participation, medal distribution, and country performance trends.

### Key Innovation

**This Azure data solution** showcases advanced cloud analytics capabilities:
- **Multi-Source Data Integration**: Automated ingestion from GitHub raw URLs using Azure Data Factory with dynamic pipeline orchestration
- **Spark-Based Transformation**: Azure Databricks processing with type correction, aggregation, and statistical calculations for 11,000+ athlete records
- **Interactive Analytics**: Comprehensive Tableau dashboard with drill-down capabilities, geographical visualization, and gender distribution analysis
- **Cloud-Native Architecture**: Complete Azure ecosystem integration from storage to visualization with scalable data processing

### Technical Architecture

The system implements a modern data lakehouse pattern with automated cloud orchestration:
- **Ingestion Layer**: Azure Data Factory with GitHub linked services and automated copy activities
- **Storage Layer**: Azure Data Lake Storage Gen2 with hierarchical namespace for raw and transformed data containers
- **Processing Layer**: Azure Databricks with Apache Spark for distributed data transformation and statistical analysis
- **Analytics Layer**: Azure Synapse Analytics with SQL querying and Power BI integration capabilities

**Technology Stack:**
- **Orchestration**: Azure Data Factory with linked services and automated pipeline scheduling
- **Storage**: Azure Data Lake Storage Gen2 with container-based data organization and access control
- **Processing**: Azure Databricks with PySpark, DataFrame operations, and mount point integration
- **Analytics**: Azure Synapse Analytics with SQL pools and external table connectivity
- **Visualization**: Tableau Public with interactive dashboards, geographical mapping, and multi-tab storytelling

### Performance & Results

The pipeline processes comprehensive Olympic datasets with optimized analytics performance:
- **Data Volume**: Successfully processed 11,000+ athlete records across 47 disciplines with 207 participating countries
- **Transformation Efficiency**: Automated data type corrections, percentage calculations, and aggregation operations using Spark distributed computing
- **Dashboard Insights**: Interactive visualization revealing medal efficiency patterns, gender distribution trends, and country participation analytics

### Implementation Highlights

- **End-to-End Automation**: Complete data pipeline from GitHub source ingestion to interactive dashboard deployment with minimal manual intervention
- **Advanced Data Transformation**: Databricks-based processing with statistical calculations, data type enforcement, and aggregation operations for analytics optimization
- **Multi-Dimensional Analytics**: Comprehensive dashboard design featuring geographical visualization, treemap analysis, and bubble charts for complex data storytelling
- **Cloud Integration**: Seamless Azure service connectivity with Data Lake mounting, Synapse integration, and external visualization platform connectivity

*This project demonstrates enterprise-grade Azure data engineering capabilities, combining automated cloud orchestration with advanced analytics visualization for comprehensive Olympic data insights.*`,
    tech_stack: ['Azure Data Factory', 'Azure Databricks', 'Azure Data Lake Gen2', 'Synapse Analytics'],
    github_url: 'https://github.com/Jafri115/olymic-azure-data-engineering-project',
    demo_url: null,
    image_name: 'azureolympics-databricks-synapse-analytics-etl.png',
    created_date: '2025-06-23',
    featured: false,
    challenges: '',
    results: ''
},
{
    slug: 'azurebi-customer-sales-analytics-pipeline',
    title: 'AzureBI – Customer & Sales Analytics Pipeline',
    description: `Built a gold-layer analytics solution by ingesting on-prem SQL Server data into Azure Data Lake Gen2, transforming in Databricks, and modeling in Synapse Analytics. Features Delta Lake storage, automated dbt tests, and CI/CD via GitHub Actions.`,
    detailed_description: `
### Project Overview

**Customer & Sales Insights Pipeline** is a comprehensive Azure-based data engineering solution that transforms on-premises SQL Server data into actionable business intelligence. The project implements a modern data lakehouse architecture with automated ETL pipelines, delivering gender-based customer analytics and product performance insights through interactive dashboards.

### Key Innovation

**This Azure data pipeline** demonstrates enterprise-grade cloud data engineering:
- **Dynamic Data Ingestion**: Self-hosted integration runtime with automated table discovery using Lookup and ForEach activities
- **Medallion Architecture**: Bronze-Silver-Gold data layers implementing progressive data refinement and quality enhancement
- **Real-time Analytics**: Synapse Analytics integration with external tables for high-performance querying of Delta Lake data
- **Security-First Design**: Azure Key Vault integration with Microsoft Entra ID for comprehensive secrets management and RBAC

### Technical Architecture

The solution implements a modern data lakehouse architecture with automated orchestration:
- **Ingestion Layer**: Azure Data Factory with self-hosted integration runtime for hybrid connectivity
- **Storage Layer**: Azure Data Lake Storage Gen2 with hierarchical namespace and Delta Lake format
- **Processing Layer**: Azure Databricks with Apache Spark for distributed data transformation
- **Analytics Layer**: Azure Synapse Analytics with external tables and SQL pools

**Technology Stack:**
- **Orchestration**: Azure Data Factory with dynamic pipeline activities and dependency management
- **Storage**: Azure Data Lake Storage Gen2 with bronze/silver/gold container architecture
- **Processing**: Azure Databricks with PySpark, Delta Lake, and automated data quality checks
- **Analytics**: Azure Synapse Analytics with external tables and SQL querying capabilities
- **Visualization**: Tableau with direct cloud connectivity and interactive filtering
- **Security**: Azure Key Vault, Microsoft Entra ID with role-based access control

### Performance & Results

The pipeline processes AdventureWorksLT2022 data with optimized performance and data quality:
- **Data Quality**: Automated deduplication using primary keys and ModifiedDate with schema enforcement
- **Processing Efficiency**: Delta Lake format with optimized partitioning and schema evolution support
- **Query Performance**: External tables in Synapse Analytics enable sub-second query responses for dashboard interactions

### Implementation Highlights

- **Medallion Data Architecture**: Progressive data refinement from raw bronze to analytics-ready gold layer with comprehensive data lineage tracking
- **Advanced ETL Patterns**: Dynamic table discovery with ForEach loops, automated schema validation, and incremental data processing capabilities
- **Cloud-Native Security**: Integrated Azure Key Vault for credential management with Microsoft Entra ID authentication and fine-grained access controls
- **Production Monitoring**: Built-in Azure monitoring with automated alerting for pipeline failures and data quality issues

*This project showcases modern Azure data engineering practices, demonstrating scalable cloud architecture design, automated data pipeline orchestration, and enterprise-grade security implementation.*`,
    tech_stack: ['Azure Data Factory', 'Azure Data Lake Gen2', 'Azure Databricks', 'Synapse Analytics', 'SQL', 'dbt'],
    github_url: 'https://github.com/Jafri115/end-to-end-data-engineering-adventure-work',
    demo_url: null,
    image_name: 'azurebi-customer-sales-analytics-pipeline.png',
    created_date: '2025-06-23',
    featured: false,
    challenges: '',
    results: ''
},
{
    slug: 'exampredict-mldriven-student-performance-forecast',
    title: 'ExamPredict – ML‑Driven Student Performance Forecast',
    description: `Predictive analytics platform leveraging Random Forest and Gradient Boosting models to forecast student exam outcomes. Features a Flask frontend, AWS Elastic Beanstalk deployment, and CI/CD pipelines using AWS CodePipeline.`,
    detailed_description: `*Machine Learning Web Application with AWS Deployment*

### Project Overview

**ExamPredict** is a comprehensive machine learning application that predicts mathematics exam scores based on demographic and educational factors. The project demonstrates end-to-end ML workflow from exploratory data analysis to production deployment, featuring automated model selection, web-based prediction interface, and cloud deployment on AWS infrastructure.

### Key Innovation

**This ML application** combines predictive analytics with production-ready deployment:
- **Automated Model Selection**: Comparative evaluation of Random Forest, Gradient Boosting, CatBoost, and XGBoost with R² scoring optimization
- **Intelligent Preprocessing Pipeline**: Automated categorical encoding and numerical scaling with persistent artifact management
- **Production Web Interface**: Flask-based application with user-friendly form inputs and real-time prediction capabilities
- **Cloud-Native Deployment**: AWS Elastic Beanstalk with CodePipeline for continuous integration and automated deployment

### Technical Architecture

The system implements a modular ML pipeline with cloud deployment automation:
- **Data Layer**: Automated ingestion with train/test splitting and artifact persistence
- **Processing Layer**: Sklearn-based preprocessing with categorical encoding and feature scaling
- **Model Layer**: Multi-algorithm training with automated hyperparameter optimization
- **Application Layer**: Flask web framework with HTML templates and form validation

**Technology Stack:**
- **Machine Learning**: Scikit-learn, CatBoost, XGBoost with automated model comparison and selection
- **Web Framework**: Flask with Jinja2 templating and responsive HTML/CSS interface design
- **Data Processing**: Pandas, NumPy for data manipulation with automated feature engineering
- **Cloud Infrastructure**: AWS Elastic Beanstalk, CodePipeline, S3 for continuous deployment automation
- **Development**: Jupyter Notebooks for EDA, modular Python architecture with setup.py configuration

### Performance & Results

The model achieves robust predictive performance with comprehensive evaluation metrics:
- **Model Accuracy**: R² scoring optimization across multiple algorithms with cross-validation for reliable performance assessment
- **Feature Engineering**: Automated preprocessing pipeline handling categorical variables and numerical scaling for optimal model input
- **Production Performance**: Sub-second prediction response times through optimized model artifacts and efficient web serving

### Implementation Highlights

- **Modular Architecture**: Separation of data ingestion, transformation, and training components with reusable preprocessing artifacts
- **Automated CI/CD Pipeline**: AWS CodePipeline integration with GitHub source control and automated Elastic Beanstalk deployment
- **Production Monitoring**: Comprehensive logging framework with artifact versioning and model performance tracking capabilities
- **Scalable Deployment**: Cloud-native architecture with IAM role management and environment variable configuration for production security

*This project demonstrates complete ML lifecycle management, from exploratory analysis to production deployment, showcasing modern MLOps practices with cloud infrastructure automation.*`,
    tech_stack: ['Python', 'Scikit-learn', 'Flask', 'AWS Elastic Beanstalk', 'AWS CodePipeline'],
    github_url: 'https://github.com/Jafri115/student-performance-ml-aws',
    demo_url: null,
    image_name: 'exampredict-ml-driven-student-performance-forecast.png',
    created_date: '2025-06-23',
    featured: false,
    challenges: '',
    results: ''
},
{
    slug: 'fraud-detection-with-transformers-and-gans',
    title: 'Fraud Detection with Transformers and GANs',
    description: `Hybrid anomaly detection system using time-aware Transformers and adversarial networks to flag fraudulent patterns in sequential data. Developed and trained a SeqTab-OCAN model, implemented data pipeline with TensorFlow and custom preprocessing, and achieved state-of-the-art results on internal and public datasets.`,
    detailed_description: `*Master's Thesis Project - University of Hildesheim*

### Project Overview
Fraudulent behaviors present a constantly evolving challenge in digital systems. Traditional detection methods often analyze transactional data or user sequences in isolation, missing the rich insights that come from combining both approaches. This project introduces **SeqTab-OCAN**, an innovative fraud detection framework that bridges this gap.

### Key Innovation
**SeqTab-OCAN** combines:
- **Sequential Data Analysis**: Time-aware attention networks capture temporal patterns in user behavior
- **Tabular Feature Integration**: Traditional fraud indicators merged with behavioral sequences  
- **Advanced GAN Architecture**: One-Class Adversarial Networks (OCAN) handle imbalanced datasets and detect subtle anomalies

### Technical Architecture

#### Core Components
- **Time-Aware Transformers**: Capture temporal dependencies in user action sequences
- **Representation Learning**: Unified embedding space for sequential and tabular features
- **Adversarial Training**: GAN-based approach for robust anomaly detection
- **Multi-Phase Pipeline**: Separate representation learning and fraud detection phases

#### Technology Stack
- **Deep Learning**: PyTorch, TensorFlow
- **Experiment Tracking**: MLflow
- **Data Processing**: Pandas, NumPy
- **Evaluation**: Scikit-learn metrics suite

### Performance Results
Extensive evaluation on the UMDWikipedia dataset demonstrates superior performance:

| Method | Precision | Recall | F1-Score | AUC-PR | AUC-ROC |
|--------|-----------|---------|----------|--------|---------|
| OCAN Baseline | 0.9117 | 0.9097 | 0.9107 | 0.8838 | 0.971 |
| Tab-RL | 0.9042 | 0.7996 | 0.8487 | 0.9240 | 0.9079 |
| Seq-RL | 0.9470 | 0.9026 | 0.9241 | 0.9718 | 0.9754 |
| **SeqTab-OCAN** | **0.9307** | **0.9487** | **0.9396** | **0.9817** | **0.9379** |

### Dataset Engineering
**Challenge**: Limited availability of datasets containing both sequential and tabular fraud indicators.

**Solution**: Engineered comprehensive features from Wikipedia vandalism data:

#### Behavioral Features
- **Activity Patterns**: Edit frequency, time-of-day preferences, weekend activity
- **Engagement Metrics**: Total edits, unique pages, page category diversity  
- **Quality Indicators**: Revert ratios, ClueBot interventions
- **Temporal Dynamics**: Sequential edit patterns and timing analysis

### Real-World Applications
This framework addresses critical challenges in:
- **Financial Services**: Credit card fraud, account takeover detection
- **E-commerce**: Fake reviews, fraudulent transactions
- **Social Platforms**: Spam detection, bot identification
- **Digital Identity**: Account verification, behavioral biometrics

### Academic Impact
This research contributes to the fraud detection field by:
- Demonstrating the value of multi-modal data fusion
- Introducing time-aware attention for sequential fraud detection
- Providing a robust framework for imbalanced dataset challenges
- Establishing new benchmarks for combined sequential-tabular approaches

### Implementation Highlights
The project features a complete ML pipeline:
- **Modular Architecture**: Separate components for easy experimentation
- **Experiment Tracking**: Comprehensive MLflow integration
- **Reproducible Results**: Detailed configuration and seed management
- **Scalable Design**: Efficient data processing and model training

---
*This project represents advanced research in applying deep learning to fraud detection, with practical implications for securing digital platforms and financial systems.*`,
    tech_stack: ['Python', 'MLFlow', 'TensorFlow', 'Transformers', 'GANs', 'Snowflake'],
    github_url: 'https://github.com/Jafri115/dynamic_fraud_detection',
    demo_url: null,
    image_name: 'fraud-detection-with-transformers-and-gans.png',
    created_date: '2025-06-23',
    featured: true,
    challenges: '',
    results: ''
},
{
    slug: 'churnguard-customer-churn-prediction-system',
    title: 'ChurnGuard – Customer Churn Prediction System',
    description: `End-to-end machine learning pipeline to predict customer churn using Logistic Regression, Random Forest, and XGBoost classifiers. Includes detailed EDA, preprocessing with imputation and encoding, feature selection, hyperparameter tuning, and evaluation with classification metrics. Interactive visualizations created using Plotly for business insight and model interpretation.`,
    detailed_description: `*End-to-End MLOps Demonstration Project*

### Project Overview

**ChurnGuard – Customer Churn Prediction System** is a comprehensive MLOps pipeline that predicts customer churn probability using advanced machine learning techniques. The project demonstrates complete ML lifecycle management from data ingestion to production deployment, featuring automated CI/CD pipelines, model monitoring, and a user-facing web application for real-time predictions.

### Key Innovation

**This MLOps project** showcases production-ready machine learning practices:
- **Advanced Feature Engineering**: Automated creation of tenure buckets, service counts, and interaction terms for enhanced model performance
- **Multi-Model Ensemble**: XGBoost/LightGBM with Logistic Regression comparison using cross-validation and AUC optimization
- **Complete CI/CD Integration**: GitLab pipelines with automated testing, Docker containerization, and deployment orchestration
- **Production Monitoring**: Prometheus metrics collection with Grafana dashboards for real-time model performance tracking

### Technical Architecture

The system implements a microservices architecture with separate ML pipeline and web application components:
- **ML Pipeline**: FastAPI service with trained model serving and MLflow experiment tracking
- **Web Application**: Flask-based user interface for interactive churn predictions
- **Monitoring Stack**: Prometheus + Grafana for operational insights and model drift detection

**Technology Stack:**
- **Machine Learning**: XGBoost, LightGBM, Scikit-learn with automated hyperparameter tuning
- **MLOps**: MLflow for experiment tracking, DVC for data versioning, automated retraining capabilities  
- **Backend**: FastAPI for model serving, Flask for web interface, Docker containerization
- **Infrastructure**: Kubernetes deployment, Azure Container Apps, GitLab CI/CD pipelines
- **Monitoring**: Prometheus metrics, Grafana dashboards, comprehensive logging framework

### Performance & Results

The model achieves robust predictive performance with production-ready metrics:
- **Model Accuracy**: Optimized using ROC-AUC scoring with cross-validation for reliable performance estimates
- **Feature Engineering Impact**: Custom feature creation improved model performance through tenure bucketing and service interaction analysis
- **Automated Pipeline**: Complete CI/CD workflow from data ingestion to deployment with zero-downtime updates

### Implementation Highlights

- **MLflow Integration**: Comprehensive experiment tracking with automated model versioning and artifact management for reproducible results
- **Production-Ready API**: FastAPI service with automatic data preprocessing, error handling, and standardized response formats
- **Advanced Data Processing**: Automated handling of class imbalance, feature scaling, and categorical encoding with persistent preprocessor artifacts
- **Container Orchestration**: Docker-based deployment with Kubernetes support and environment-specific configuration management

*This project demonstrates enterprise-level MLOps capabilities, combining modern ML practices with robust software engineering principles for scalable, maintainable machine learning systems.*`,
    tech_stack: ['Python', 'Pandas', 'Scikit-learn', 'XGBoost', 'Plotly', 'Matplotlib', 'Seaborn'],
    github_url: 'https://github.com/Jafri115/customer-churn-prediction',
    demo_url: null,
    image_name: 'churnguard-customer-churn-prediction-system.png',
    created_date: '2025-06-23',
    featured: false,
    challenges: '',
    results: ''
},
{
    slug: 'dissolve-secure-selferasing-data-platform',
    title: 'DISSOLVE – Secure Self‑Erasing Data Platform',
    description: `A Java-based secure data storage system that encrypts files at rest and supports self-destructing records. Implements symmetric AES encryption, key rotation, and RESTful API endpoints for secure client-server interactions.`,
    detailed_description: ` *Academic Research Project - NED University of Engineering & Technology*

### Project Overview

**DISSOLVE** is an innovative self-destructive data system that addresses the critical problem of permanent data persistence online. The system automatically makes sensitive emails, SMS messages, and files completely unreadable after user-specified time periods, ensuring true data privacy and control through dynamic password systems and AES encryption.

### Key Innovation

**DISSOLVE** combines cutting-edge security technologies to solve the "Internet never forgets" problem:
- **Dynamic One-Time Passwords (OTP)**: Centralized key generation with automatic destruction after expiration
- **AES Encryption (128/192/256-bit)**: Industry-standard encryption with zero server storage of confidential data
- **Multi-Platform Architecture**: Seamless integration across web applications and Android mobile devices
- **Irreversible Data Destruction**: No recovery possible once expired - even for system administrators

### Technical Architecture

The system employs a distributed client-server architecture with three core components:
- **Client Applications**: Web interface for files/emails and Android app for SMS
- **DISSOLVE Server**: Central processing hub with SQLite database
- **OTP Generator**: Secure web service for dynamic key management

**Technology Stack:**
- **Backend**: Python, Django Framework, RESTful APIs
- **Database**: SQLite with custom schema design
- **Mobile**: Android Studio, Java, QPython3
- **Security**: AES encryption, HTTPS/TLS, secure key exchange protocols
- **Frontend**: HTML/CSS/JavaScript with responsive design

### Performance & Security Results

The system demonstrates robust security measures with **zero permanent data storage** on servers:
- **Encryption Strength**: NIST-approved AES implementation resistant to brute force attacks
- **Key Management**: Dynamic key generation eliminates long-term storage vulnerabilities
- **Multi-Channel Support**: Handles emails, SMS, and file sharing with consistent security protocols

### Implementation Highlights

- **Microservices Architecture**: Separation of concerns between OTP generation, data processing, and client interfaces
- **Cross-Platform Development**: Unified codebase supporting both web and mobile platforms with shared security protocols
- **Comprehensive Testing Strategy**: Unit, integration, system, and security testing with automated test suites
- **Security-First Design**: Implemented threat modeling and audit trails while maintaining user privacy through minimal logging

*This project demonstrates advanced full-stack development capabilities, modern security practices, and innovative approaches to data privacy challenges in distributed systems.*`,
    tech_stack: ['Java', 'Spring Boot', 'JUnit', 'REST API', 'Encryption'],
    github_url: undefined,
    demo_url: null,
    image_name: 'dissolve-secure-self-erasing-data-platform.png',
    created_date: '2025-06-23',
    featured: false,
    challenges: '',
    results: ''
}
];