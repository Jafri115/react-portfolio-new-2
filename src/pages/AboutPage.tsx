// src/pages/AboutPage.tsx
import React from 'react';
import { scroller } from 'react-scroll';
import Card from '../components/Card';
import { experiences } from '../data/experiences';
import { skills, skills_categories } from '../data/skills';
import styles from "./AboutPage.module.css";

const AboutPage: React.FC = () => {
    const skillsByCategory = skills.reduce((acc, skill) => {
        (acc[skill.category] = acc[skill.category] || []).push(skill);
        return acc;
    }, {} as Record<string, typeof skills>);

    const handleScrollTo = (elementId: string) => {
        scroller.scrollTo(elementId, {
            duration: 500,
            delay: 0,
            smooth: 'easeInOutQuart',
            offset: -80,
        });
    };

    return (
        <div className={styles.aboutPage}>
            {/* ... Rest of the JSX from your AboutPage.jsx ... */}
            {/* The important part is adding the React.FC type and typing the handler */}
            <div className={styles.buttonGroup}>
                <a onClick={() => handleScrollTo('experience')} className="btn btn-primary" style={{cursor: 'pointer'}}>View Experience</a>
                <a onClick={() => handleScrollTo('skills')} className="btn btn-secondary" style={{cursor: 'pointer'}}>See Skills</a>
            </div>
            {/* ... etc ... */}
        </div>
    );
};

export default AboutPage;