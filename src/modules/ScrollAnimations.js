/**
 * ScrollAnimations — GSAP ScrollTrigger setup for all sections.
 */
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export function initScrollAnimations() {
    // ─── REVEAL ANIMATIONS ───
    // .reveal-up
    gsap.utils.toArray('.reveal-up').forEach((el) => {
        gsap.fromTo(el,
            { y: 40, opacity: 0 },
            {
                y: 0,
                opacity: 1,
                duration: 0.8,
                ease: 'power3.out',
                scrollTrigger: {
                    trigger: el,
                    start: 'top 88%',
                    toggleActions: 'play none none none',
                },
            }
        );
    });

    // .reveal-left
    gsap.utils.toArray('.reveal-left').forEach((el) => {
        gsap.fromTo(el,
            { x: -50, opacity: 0 },
            {
                x: 0,
                opacity: 1,
                duration: 0.8,
                ease: 'power3.out',
                scrollTrigger: {
                    trigger: el,
                    start: 'top 88%',
                    toggleActions: 'play none none none',
                },
            }
        );
    });

    // .reveal-right
    gsap.utils.toArray('.reveal-right').forEach((el) => {
        gsap.fromTo(el,
            { x: 50, opacity: 0 },
            {
                x: 0,
                opacity: 1,
                duration: 0.8,
                ease: 'power3.out',
                scrollTrigger: {
                    trigger: el,
                    start: 'top 88%',
                    toggleActions: 'play none none none',
                },
            }
        );
    });

    // ─── HERO INTRO STAGGER ───
    const heroElements = document.querySelectorAll('#home .reveal-up');
    if (heroElements.length) {
        gsap.fromTo(heroElements,
            { y: 50, opacity: 0 },
            {
                y: 0,
                opacity: 1,
                duration: 0.9,
                stagger: 0.12,
                ease: 'power3.out',
                delay: 0.3,
            }
        );
    }

    // ─── NAVBAR GLASS ON SCROLL ───
    const navbar = document.getElementById('navbar');
    if (navbar) {
        ScrollTrigger.create({
            start: 'top -80',
            onUpdate: (self) => {
                if (self.scroll() > 80) {
                    navbar.classList.add('bg-deep-space/80', 'backdrop-blur-xl', 'border-b', 'border-white/[0.05]', 'shadow-glass');
                } else {
                    navbar.classList.remove('bg-deep-space/80', 'backdrop-blur-xl', 'border-b', 'border-white/[0.05]', 'shadow-glass');
                }
            },
        });
    }

    // ─── SKILL BARS ANIMATION ───
    const skillBars = document.querySelectorAll('.skill-bar');
    skillBars.forEach((bar) => {
        const fill = bar.querySelector('.progress-fill');
        const percentSpan = bar.querySelector('.skill-percent');
        const target = parseInt(bar.dataset.percent, 10) || 0;

        ScrollTrigger.create({
            trigger: bar,
            start: 'top 90%',
            onEnter: () => {
                fill?.classList.add('animate');
                // Animate counter
                if (percentSpan) {
                    gsap.to({ val: 0 }, {
                        val: target,
                        duration: 1.2,
                        ease: 'power2.out',
                        onUpdate: function () {
                            percentSpan.textContent = Math.round(this.targets()[0].val) + '%';
                        },
                    });
                }
            },
        });
    });

    // ─── RADIAL SKILLS ANIMATION ───
    const radialSkills = document.querySelectorAll('.radial-skill');
    radialSkills.forEach((el) => {
        const circle = el.querySelector('.radial-fill');
        const percentSpan = el.querySelector('.radial-percent');
        const target = parseInt(el.dataset.percent, 10) || 0;
        const circumference = Math.round(2 * Math.PI * 80); // r=80 → ~502

        // Set initial SVG attributes (circle starts fully hidden)
        if (circle) {
            circle.setAttribute('stroke-dasharray', circumference);
            circle.setAttribute('stroke-dashoffset', circumference);
        }

        const targetOffset = circumference - (circumference * target / 100);

        ScrollTrigger.create({
            trigger: el,
            start: 'top 90%',
            onEnter: () => {
                if (circle) {
                    gsap.to(circle, {
                        attr: { 'stroke-dashoffset': targetOffset },
                        duration: 1.5,
                        ease: 'power2.out',
                    });
                }
                if (percentSpan) {
                    gsap.to({ val: 0 }, {
                        val: target,
                        duration: 1.5,
                        ease: 'power2.out',
                        onUpdate: function () {
                            percentSpan.textContent = Math.round(this.targets()[0].val) + '%';
                        },
                    });
                }
            },
        });
    });

    // ─── PORTFOLIO 3D TILT ───
    const portfolioCards = document.querySelectorAll('.portfolio-card');
    portfolioCards.forEach((card) => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -6;
            const rotateY = ((x - centerX) / centerX) * 6;

            gsap.to(card, {
                rotateX,
                rotateY,
                transformPerspective: 800,
                duration: 0.4,
                ease: 'power2.out',
            });
        });

        card.addEventListener('mouseleave', () => {
            gsap.to(card, {
                rotateX: 0,
                rotateY: 0,
                duration: 0.6,
                ease: 'power2.out',
            });
        });
    });

    // ─── ABOUT IMAGE TILT ───
    const aboutImage = document.getElementById('about-image');
    if (aboutImage) {
        aboutImage.addEventListener('mousemove', (e) => {
            const rect = aboutImage.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -8;
            const rotateY = ((x - centerX) / centerX) * 8;

            gsap.to(aboutImage, {
                rotateX,
                rotateY,
                transformPerspective: 800,
                duration: 0.4,
                ease: 'power2.out',
            });
        });

        aboutImage.addEventListener('mouseleave', () => {
            gsap.to(aboutImage, {
                rotateX: 0,
                rotateY: 0,
                duration: 0.7,
                ease: 'elastic.out(1, 0.5)',
            });
        });
    }
}
