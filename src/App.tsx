// src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';
import { ProjectsPage } from './pages/ProjectsPage';
// We have commented out the pages that do not exist yet.
// import ProjectDetailPage from './pages/ProjectDetailPage';
// import BlogPage from './pages/BlogPage';
// import BlogDetailPage from './pages/BlogDetailPage';
import './assets/css/global.css';

const ScrollToTop = () => {
  const { pathname } = useLocation();
  React.useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
};

const App: React.FC = () => {
  return (
    <Router>
      <ScrollToTop />
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          {/* Also comment out the routes for non-existent pages */}
          {/* <Route path="/projects/:slug" element={<ProjectDetailPage />} /> */}
          {/* <Route path="/blog" element={<BlogPage />} /> */}
          {/* <Route path="/blog/:slug" element={<BlogDetailPage />} /> */}
        </Routes>
      </Layout>
    </Router>
  );
};

export default App;