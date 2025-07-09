import type { Project, SkillCategory } from '../types';

export const projects: Project[] = [
  {
    id: 1,
    title: 'Semantic Book Recommender',
    category: 'AI / Machine Learning',
    description: 'A semantic search system using LLMs and vector search (ChromaDB) to find books based on natural language queries, sentiment, and emotion analysis.',
    imageUrl: '/images/project-recommender.png',
    tech: ['Python', 'LangChain', 'OpenAI', 'ChromaDB', 'Gradio'],
    liveUrl: 'https://huggingface.co/spaces/Wasifjafri/semantic-book-recommender',
    githubUrl: 'https://github.com/Jafri115/llm-book-recommender',
  },
  {
    id: 2,
    title: 'Fraud Detection with Transformers',
    category: 'AI Research (Thesis)',
    description: 'A hybrid anomaly detection system using time-aware Transformers and GANs to flag fraudulent patterns in sequential and tabular data.',
    imageUrl: '/images/project-fraud.png',
    tech: ['Python', 'MLFlow', 'TensorFlow', 'Transformers', 'GANs'],
    githubUrl: 'https://github.com/Jafri115/dynamic_fraud_detection',
  },
  {
    id: 3,
    title: 'Cloud Data Orchestrator',
    category: 'Data Engineering',
    description: 'A scalable ELT pipeline using Snowflake, dbt for transformations, and Airflow for workflow orchestration and automated testing.',
    imageUrl: '/images/project-pipeline.png',
    tech: ['Snowflake', 'dbt', 'Airflow', 'SQL', 'Docker'],
    githubUrl: 'https://github.com/Jafri115/snowflake-dbt-airflow-elt-pipeline',
  },
  {
    id: 4,
    title: 'StyleSense Fashion Classifier',
    category: 'Web & Machine Learning',
    description: 'A Flask web app for real-time fashion image classification using TensorFlow and Hugging Face models, deployed with Docker.',
    imageUrl: '/images/project-stylesense.png',
    tech: ['Python', 'Flask', 'TensorFlow', 'Hugging Face', 'Docker'],
    githubUrl: 'https://github.com/Jafri115/huggingface-resnet-fmnist',
  },
];

export const skills: SkillCategory[] = [
  {
    title: 'Data & Machine Learning',
    skills: ['Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy', 'LangChain', 'Transformers', 'GANs'],
  },
  {
    title: 'Data Engineering',
    skills: ['SQL', 'PySpark', 'Airflow', 'dbt', 'Snowflake', 'PostgreSQL', 'Docker'],
  },
  {
    title: 'Web & Cloud',
    skills: ['React', 'TypeScript', 'Node.js', 'FastAPI', 'Flask', 'AWS', 'Azure', 'Kubernetes'],
  },
  {
    title: 'Tools & DevOps',
    skills: ['Git', 'CI/CD', 'MLflow', 'Tableau', 'Gradio'],
  },];
