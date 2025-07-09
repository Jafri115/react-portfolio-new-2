import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { HelmetProvider } from 'react-helmet-async';
import App from './App.tsx';
import './index.css';
import { useScrollToTop } from './hooks/useScrollToTop.ts';

const AppWrapper = () => {
  useScrollToTop();
  return <App />;
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HelmetProvider>
      <BrowserRouter>
        <AppWrapper />
      </BrowserRouter>
    </HelmetProvider>
  </React.StrictMode>
);