// src/components/MainLayout.tsx
import Header from './Header';
import Footer from './Footer';
import type { ReactNode } from 'react';

export const MainLayout = ({ children }: { children: ReactNode }) => {
  return (
    // ADDED: flex flex-col to stack header, main, and footer vertically
    <div className="flex flex-col min-h-screen">
      <Header />
      {/* ADDED: flex-grow to make the main content take up all available space */}
      <main className="flex-grow">{children}</main>
      <Footer />
    </div>


  );};

