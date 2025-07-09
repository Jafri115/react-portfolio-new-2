// src/components/Card.tsx
import React from 'react';
import { motion } from 'framer-motion';
import styles from './Card.module.css';

const cardVariants = {
  offscreen: { y: 20, opacity: 0 },
  onscreen: {
    y: 0,
    opacity: 1,
    transition: { type: "spring", bounce: 0.4, duration: 0.8 }
  }
};

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hoverEffect?: boolean;
}

const Card: React.FC<CardProps> = ({ children, className = '', hoverEffect = false }) => {
  const combinedClassName = [
    styles.card,
    hoverEffect && styles.cardHover,
    className
  ].filter(Boolean).join(' ');

  return (
    <motion.div
      className={combinedClassName}
      initial="offscreen"
      whileInView="onscreen"
      viewport={{ once: true, amount: 0.1 }}
      variants={cardVariants}
    >
      {children}
    </motion.div>
  );
};

export default Card;