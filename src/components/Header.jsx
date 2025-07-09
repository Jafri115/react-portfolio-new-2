import React, { useState, useEffect } from 'react';
import { NavLink, Link } from 'react-router-dom';
import styles from './Header.module.css';

const Header = () => {
    const [isScrolled, setIsScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            if (window.scrollY > 50) {
                setIsScrolled(true);
            } else {
                setIsScrolled(false);
            }
        };

        window.addEventListener('scroll', handleScroll);
        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, []);

    const getLinkClass = ({ isActive }) => {
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
                    {/* <li><NavLink to="/blog" className={getLinkClass}>blog</NavLink></li> */}
                </ul>
            </nav>
        </header>
    );
};

export default Header;