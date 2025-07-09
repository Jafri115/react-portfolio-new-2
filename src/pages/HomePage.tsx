// src/pages/HomePage.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { FaFolderOpen, FaUser, FaGithub, FaCalendar, FaArrowRight } from 'react-icons/fa';
import Card from '../components/Card';
import Terminal from '../components/Terminal';
import { projects } from '../data/projects';
import styles from './HomePage.module.css';
import StatsSection from '../components/StatsSection';
import TechStack from '../components/TechStack';

const HomePage: React.FC = () => {
    const featuredProjects = projects.filter(p => p.featured).slice(0, 4);

    return (
        <div className={styles.homePage}>
            {/* --- Hero Section --- */}
            <section className={styles.heroSection}>
                <div className={styles.heroContent}>
                    <h1 className="h1-gradient">Data Scientist & ML Engineer</h1>
                    <p className={styles.subtitle}>
                        Transforming raw data into actionable insights for evidence-based decisions
                    </p>
                    <Terminal />
                    <div className={styles.buttonGroup}>
                        <Link to="/projects" className="btn btn-primary"><FaFolderOpen /> View Projects</Link>
                        <Link to="/about" className="btn btn-secondary"><FaUser /> About Me</Link>
                        <a href="https://github.com/Jafri115" target="_blank" rel="noopener noreferrer" className="btn btn-secondary"><FaGithub /> GitHub</a>
                    </div>
                </div>
                <div className={styles.heroInfoCard}>
                    <Card>
                        <div className={styles.initialsCircle}>WJ</div>
                        <h3 className={styles.infoName}>Wasif Jafri</h3>
                        <p className={styles.infoTitle}>Data Scientist & ML Engineer</p>
                        <p className={styles.infoDetails}>
                            🏢 Ionos (Former)<br />
                            📍 Hildesheim, Germany<br />
                            🎓 MSc Data Analytics
                        </p>
                    </Card>
                </div>
            </section>

            {/* --- Stats Section --- */}
            <StatsSection />

            {/* --- Tech Stack Section --- */}
            <TechStack />

            {/* --- Featured Projects Section --- */}
            <section className={styles.featuredSection}>
                <h2 className={styles.sectionTitle}>Featured Projects</h2>
                <div className={styles.grid}>
                    {featuredProjects.map(project => (
                        <Card key={project.slug} hoverEffect className={styles.projectCard}>
                            <div className={styles.cardContent}>
                                <div>
                                    <h3 className={styles.projectTitle}>{project.title}</h3>
                                    <p className={styles.projectDescription}>{project.description.substring(0, 120)}...</p>
                                    <div className={styles.techStack}>
                                        {project.tech_stack.split(',').slice(0, 4).map(tech => (
                                            <span key={tech} className="skill-tag">{tech.trim()}</span>
                                        ))}
                                    </div>
                                </div>
                                <div className={styles.projectFooter}>
                                    <span className={styles.projectDate}>
                                        <FaCalendar /> {new Date(project.created_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
                                    </span>
                                    <Link to={`/projects/${project.slug}`} className="btn">
                                        View Project <FaArrowRight />
                                    </Link>
                                </div>
                            </div>
                        </Card>
                    ))}
                </div>
            </section>

            {/* --- Call to Action Section --- */}
            <section className={styles.ctaSection}>
                <h2 className={styles.ctaHeading}>Let's Build Data-Driven Solutions</h2>
                <p className={styles.ctaText}>
                    Always excited to collaborate on challenging data problems and innovative ML solutions that drive social impact.
                </p>
                <div className={styles.ctaButtons}>
                    <Link to="/projects" className="btn btn-primary">View All Projects</Link>
                    <Link to="/about" className="btn btn-secondary">Get In Touch</Link>
                </div>
            </section>
        </div>
    );
};

export default HomePage;