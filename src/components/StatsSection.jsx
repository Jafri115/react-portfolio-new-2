// src/components/StatsSection.jsx
import React from 'react';
import Card from './Card';
import AnimatedCounter from './AnimatedCounter';
import styles from './StatsSection.module.css';

const stats = [
    { value: 5, text: "+", label: "Years Experience" },
    { value: 85, text: "%", label: "ETL Runtime Reduction" },
    { value: 11, text: "%", label: "F1 Score Improvement" },
    { value: 1.4, text: "", label: "Masters CGPA" ,decimal: true}
];

const StatsSection = () => {
  return (
    <section className={styles.statsGrid}>
      {stats.map((stat, index) => (
        <Card key={index} className={styles.statCard}>
          <div className={styles.statNumber}>
            {/* Here we use the new animated counter */}
            <AnimatedCounter value={stat.value} decimal={stat.decimal} />
            {stat.text}
          </div>
          <div className={styles.statLabel}>{stat.label}</div>
        </Card>
      ))}
    </section>
  );
};

export default StatsSection;