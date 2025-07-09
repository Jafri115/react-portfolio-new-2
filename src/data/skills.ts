// src/data/skills.ts
import type { Skill } from '../types';

export const skills: Skill[] = [
  // ... Paste the content of your skills.js file here
  // Example:
  { name: 'Python', category: 'programming', proficiency: 95, icon: 'fab fa-python' },
  // ... and so on
];

export const skills_categories: Record<string, string> = {
  programming: 'Programming Languages & Libraries',
  framework: 'Frameworks & Libraries',
  database: 'Data Engineering & Pipelines',
  tool: 'Web Development & Deployment',};