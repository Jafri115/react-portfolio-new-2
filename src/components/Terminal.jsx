import React from 'react';
import styles from './Terminal.module.css';

const Terminal = () => {
  return (
    <div className={styles.terminalWindow}>
      <div className={styles.terminalHeader}>
        <div className={`${styles.terminalDot} ${styles.dotRed}`}></div>
        <div className={`${styles.terminalDot} ${styles.dotYellow}`}></div>
        <div className={`${styles.terminalDot} ${styles.dotGreen}`}></div>
        <span className={styles.title}>bash</span>
      </div>
      <div className={styles.terminalContent}>
        <div className={styles.terminalLine}>
          <span className={styles.prompt}>wasif@data-lab:~$</span>
          <span className={styles.command}>whoami</span>
        </div>
        <div className={styles.terminalLine}>
          <span className={styles.output}>Data Scientist & ML Engineer</span>
        </div>
        <div className={styles.terminalLine}>
          <span className={styles.prompt}>wasif@data-lab:~$</span>
          <span className={styles.command}>cat specialties.txt</span>
        </div>
        <div className={styles.terminalLine}><span className={styles.output}>Machine Learning & NLP</span></div>
        <div className={styles.terminalLine}><span className={styles.output}>Data Engineering & Pipelines</span></div>
        <div className={styles.terminalLine}><span className={styles.output}>Statistical Analysis & Modeling</span></div>
        <div className={styles.terminalLine}><span className={styles.output}>Backend Development & APIs</span></div>
        <div className={styles.terminalLine}>
          <span className={styles.prompt}>wasif@data-lab:~$</span>
          <span className={styles.command}>python anomaly_detection.py</span>
        </div>
        <div className={styles.terminalLine}>
          <span className={styles.output}>Loading SeqTab-OCAN model<span className={styles.typing}>_</span></span>
        </div>
      </div>
    </div>
  );
};

export default Terminal;