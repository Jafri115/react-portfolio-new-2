// src/components/Header.tsx
import React, { useState, useEffect } from 'react';
import { NavLink, Link } from 'react-router-dom';
import styles from './Header.module.css';

const Header: React.FC = () => {
    const [isScrolled, setIsScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 50);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const getLinkClass = ({ isActive }: { isActive: boolean }): string => {
        return isActive ? `${styles.navLink} ${styles.active}` : styles.navLink;
    };

    return (
        <header className={`${styles.header} ${isScrolled ? styles.scrolled : ''}`}>
            <nav className={`container ${styles.nav}`}>
                <Link to="/" className={styles.logo}>~/data-scientist</Link>
                <ul className={styles.navLinks}>
                    <li><NavLink to="/" className={getLinkClass}>home</NavLink></li>
                    <li><NavLink to="/about" className={getLinkClass}>about</NavLink></li>
                    <li><NavLink to="/projects" className={getLinkClass}>projects</NavLink></li>
                </ul>
            </nav>
        </header>
    );
};
export default Header;
