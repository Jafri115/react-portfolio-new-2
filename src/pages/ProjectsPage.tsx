import { Helmet } from 'react-helmet-async';
import type { Project } from '../types';
import { projects } from '../data/content';
import { Github, ExternalLink } from 'lucide-react';

export const ProjectsPage = () => {
  return (
    <>
      <Helmet>
        <title>Projects - Wasif Jafri</title>
        <meta name="description" content="A showcase of data science, AI, and data engineering projects by Wasif Jafri." />
      </Helmet>
      <div className="max-w-7xl mx-auto py-16 px-4">
        <h1 className="text-4xl font-bold text-center mb-12 text-white">Projects</h1>
        <div className="space-y-16">
          {projects.map((project: Project) => (
            <div key={project.id} className="grid md:grid-cols-2 gap-8 md:gap-12 items-center">
              <div className={`md:order-${project.id % 2 === 0 ? '2' : '1'}`}>
                <a href={project.liveUrl || project.githubUrl || '#'} target="_blank" rel="noopener noreferrer">
                  <img src={project.imageUrl} alt={project.title} className="rounded-lg shadow-lg shadow-black/30 transform hover:scale-105 transition-transform duration-300" />
                </a>
              </div>
              <div className={`md:order-${project.id % 2 === 0 ? '1' : '2'}`}>
                <span className="text-sm font-bold text-cyan-400">{project.category}</span>
                <h2 className="text-3xl font-bold my-2 text-white">{project.title}</h2>
                <p className="text-slate-400 mb-4">{project.description}</p>
                <div className="flex flex-wrap gap-2 mb-6">
                  {project.tech.map(t => <span key={t} className="bg-slate-700 text-xs px-2 py-1 rounded">{t}</span>)}
                </div>
                <div className="flex items-center gap-6">
                  {project.liveUrl && <a href={project.liveUrl} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 flex items-center gap-2 font-semibold">Live Demo <ExternalLink size={16} /></a>}
                  {project.githubUrl && <a href={project.githubUrl} target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white flex items-center gap-2 font-semibold">GitHub <Github size={16} /></a>}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );};
