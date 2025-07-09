import Card from './Card';
import styles from './TechStack.module.css';

const techCategories: Record<string, string[]> = {
  'Programming & Libraries': ['Python', 'SQL', 'Java', 'TypeScript', 'Pandas', 'NumPy'],
  'ML/DL & NLP': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'NLTK', 'Transformers', 'Hugging Face'],
  'Data Engineering': ['PySpark', 'Snowpark', 'DBT', 'Airflow', 'PostgreSQL', 'MLflow'],
  'Web Development & Cloud': ['FastAPI', 'Flask', 'Django', 'Docker', 'AWS', 'Azure'],
};

const TechStack = () => {
    return (
        <section className={styles.techSection}>
            <h2 className="section-title">Tech Stack</h2>
            <div className={styles.techGrid}>
                {Object.entries(techCategories).map(([category, techs]) => (
                    <Card key={category} hoverEffect>
                        <h3 className={styles.categoryTitle}>{category}</h3>
                        <div className={styles.tagContainer}>
                            {techs.map(tech => (
                                <span key={tech} className="skill-tag">{tech}</span>
                            ))}
                        </div>
                    </Card>
                ))}
            </div>
        </section>
    );
};
export default TechStack;

