// src/components/Layout.tsx
import type { ReactNode } from 'react';
import Header from './Header';
import Footer from './Footer';

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
    return (
        <>
            <Header />
            <main className="container">
                {children}
            </main>
            <Footer />
        </>
    );
};
export default Layout;
