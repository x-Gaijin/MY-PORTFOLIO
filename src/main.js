/**
 * ╔══════════════════════════════════════════════╗
 *  Tristhan Cabrera — Engineering Portfolio
 *  Main Entry Point (ES6 Module)
 * ╚══════════════════════════════════════════════╝
 */

import './styles/main.css';

// ─── Components ───
import { initNavbar } from './components/Navbar.js';
import { initChatPanel } from './components/ChatPanel.js';
import { initContactForm } from './components/ContactForm.js';

// ─── Modules ───
import { initScrollAnimations } from './modules/ScrollAnimations.js';
import { initParticleBackground } from './modules/ParticleBackground.js';

// ─── Typed.js ───
import Typed from 'typed.js';

import gsap from 'gsap';

// ────────────────────────────────────────────────
//  BOOTSTRAP
// ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Lock scroll during preloader
    document.body.classList.add('preloader-active');

    // ─── Preloader Animation Timeline ───
    const preloader = document.getElementById('preloader');
    const preloaderBar = document.getElementById('preloader-bar');

    if (preloader) {
        const tl = gsap.timeline({
            onComplete: () => {
                // Remove preloader from DOM after animation
                preloader.remove();
                document.body.classList.remove('preloader-active');

                // Now initialize everything
                initNavbar();
                initContactForm();
                initChatPanel();
                initScrollAnimations();
                initParticleBackground('particle-canvas');

                // Typed.js
                new Typed('#typed-output', {
                    strings: ['Cybersecurity Enthusiast', 'AI Specialist', 'Software Developer', 'Circuit Designer'],
                    typeSpeed: 60,
                    backSpeed: 40,
                    backDelay: 2000,
                    loop: true,
                    cursorChar: '|',
                });

                // ─── Hero content stagger reveal ───
                gsap.from('#navbar', { y: -40, opacity: 0, duration: 0.8, ease: 'power3.out' });
                gsap.from('.hero-content > *', {
                    y: 40,
                    opacity: 0,
                    duration: 0.7,
                    stagger: 0.12,
                    ease: 'power3.out',
                    delay: 0.1,
                });
            },
        });

        // Step 1: Fade in logo
        tl.to('.preloader-logo', {
            opacity: 1,
            y: 0,
            duration: 0.6,
            ease: 'power3.out',
        });

        // Step 2: Show & animate progress bar
        tl.to('.preloader-bar-track', {
            opacity: 1,
            duration: 0.3,
            ease: 'power2.out',
        }, '-=0.2');

        tl.to(preloaderBar, {
            width: '100%',
            duration: 1.2,
            ease: 'power2.inOut',
        }, '-=0.1');

        // Step 3: Fade in tagline
        tl.to('.preloader-tagline', {
            opacity: 1,
            y: 0,
            duration: 0.4,
            ease: 'power2.out',
        }, '-=0.8');

        // Step 4: Exit — scale up logo, fade everything out
        tl.to('.preloader-content', {
            scale: 1.1,
            opacity: 0,
            duration: 0.5,
            ease: 'power3.in',
            delay: 0.3,
        });

        tl.to('.preloader-grid', {
            opacity: 0,
            duration: 0.3,
        }, '-=0.3');

        tl.to(preloader, {
            opacity: 0,
            duration: 0.4,
            ease: 'power2.inOut',
        }, '-=0.2');

    } else {
        // No preloader — init directly
        initNavbar();
        initContactForm();
        initChatPanel();
        initScrollAnimations();
        initParticleBackground('particle-canvas');

        new Typed('#typed-output', {
            strings: ['Cybersecurity Enthusiast', 'AI Specialist', 'Software Developer', 'Circuit Designer'],
            typeSpeed: 60,
            backSpeed: 40,
            backDelay: 2000,
            loop: true,
            cursorChar: '|',
        });
    }

    // ─── Smooth Scroll for Anchor Links ───
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
        anchor.addEventListener('click', (e) => {
            const href = anchor.getAttribute('href');
            if (href === '#') return;
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // ─── Category Card Mouse-Follow Glow ───
    document.querySelectorAll('.category-card').forEach((card) => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            const y = ((e.clientY - rect.top) / rect.height) * 100;
            card.style.setProperty('--mouse-x', `${x}%`);
            card.style.setProperty('--mouse-y', `${y}%`);
        });
    });

    console.log('🚀 Portfolio initialized');
});
