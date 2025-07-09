import React from 'react';

const Footer = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="container">
            <p>© {currentYear} Wasif Jafri. Built with ❤️ and lots of ☕</p>
            <p style={{ marginTop: '10px' }}>
                <a href="https://github.com/jafri115" target="_blank" rel="noopener noreferrer">GitHub</a>
                <a href="https://linkedin.com/in/wasifmurtaza" target="_blank" rel="noopener noreferrer">LinkedIn</a>
                <a href="mailto:swasifmurtaza@gmail.com">Email</a>
            </p>
        </footer>
    );
};

export default Footer;