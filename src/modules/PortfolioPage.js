/**
 * PortfolioPage.js — Shared module for all category portfolio pages.
 * Renders project cards from data and provides a lightbox image carousel.
 */
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

// ────────────────────────────────────────────────
//  RENDER PROJECT CARDS
// ────────────────────────────────────────────────
export function renderProjectCards(projects, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = projects.map((project, idx) => `
        <div class="project-card reveal-up" style="animation-delay:${idx * 0.08}s" data-project-idx="${idx}">
            <div class="overflow-hidden">
                <img src="${project.thumbnail}" alt="${project.title}" class="project-card-img" loading="lazy" />
            </div>
            <div class="project-card-body">
                <h3 class="font-display text-xl font-bold text-white mb-2">${project.title}</h3>
                <p class="text-white/40 text-sm leading-relaxed mb-4">${project.description}</p>
                <div class="flex flex-wrap gap-2 mb-4">
                    ${project.tech.map(t => `<span class="tech-badge">${t}</span>`).join('')}
                </div>
                ${project.features ? `
                <div class="mb-4">
                    <h4 class="text-xs font-mono text-white/30 tracking-wider uppercase mb-2">Key Features</h4>
                    <ul class="space-y-1">
                        ${project.features.map(f => `
                            <li class="flex items-start gap-2 text-white/40 text-xs">
                                <i class='bx bx-check text-neon-cyan mt-0.5 flex-shrink-0'></i>
                                <span>${f}</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                ` : ''}
                <div class="flex items-center gap-3 mt-auto pt-2">
                    ${project.images && project.images.length > 0 ? `
                        <button class="gallery-trigger inline-flex items-center gap-2 text-neon-cyan text-xs font-medium hover:gap-3 transition-all duration-300 cursor-pointer"
                                data-images='${JSON.stringify(project.images)}'>
                            <i class='bx bx-images'></i> View Gallery (${project.images.length})
                        </button>
                    ` : ''}
                    ${project.github ? `
                        <a href="${project.github}" target="_blank" rel="noopener"
                           class="inline-flex items-center gap-1.5 text-white/40 text-xs font-medium hover:text-neon-cyan transition-colors duration-300">
                            <i class='bx bxl-github'></i> GitHub
                        </a>
                    ` : ''}
                    ${project.demo ? `
                        <a href="${project.demo}" target="_blank" rel="noopener"
                           class="inline-flex items-center gap-1.5 text-white/40 text-xs font-medium hover:text-neon-cyan transition-colors duration-300">
                            <i class='bx bx-link-external'></i> Demo
                        </a>
                    ` : ''}
                </div>
            </div>
        </div>
    `).join('');
}


// ────────────────────────────────────────────────
//  LIGHTBOX / IMAGE CAROUSEL
// ────────────────────────────────────────────────
let lightboxEl = null;
let currentImages = [];
let currentIndex = 0;

function createLightbox() {
    if (lightboxEl) return;

    lightboxEl = document.createElement('div');
    lightboxEl.className = 'lightbox-overlay';
    lightboxEl.id = 'lightbox';
    lightboxEl.innerHTML = `
        <button class="lightbox-close" id="lb-close"><i class='bx bx-x'></i></button>
        <button class="lightbox-btn lightbox-btn--prev" id="lb-prev"><i class='bx bx-chevron-left'></i></button>
        <button class="lightbox-btn lightbox-btn--next" id="lb-next"><i class='bx bx-chevron-right'></i></button>
        <img class="lightbox-img" id="lb-img" src="" alt="Gallery Image" />
        <div class="lightbox-counter" id="lb-counter"></div>
        <div class="lightbox-dots" id="lb-dots"></div>
    `;
    document.body.appendChild(lightboxEl);

    // Close
    document.getElementById('lb-close').addEventListener('click', closeLightbox);
    lightboxEl.addEventListener('click', (e) => {
        if (e.target === lightboxEl) closeLightbox();
    });

    // Nav
    document.getElementById('lb-prev').addEventListener('click', () => navigateLightbox(-1));
    document.getElementById('lb-next').addEventListener('click', () => navigateLightbox(1));

    // Keyboard
    document.addEventListener('keydown', (e) => {
        if (!lightboxEl.classList.contains('active')) return;
        if (e.key === 'Escape') closeLightbox();
        if (e.key === 'ArrowLeft') navigateLightbox(-1);
        if (e.key === 'ArrowRight') navigateLightbox(1);
    });
}

function openLightbox(images, startIdx = 0) {
    createLightbox();
    currentImages = images;
    currentIndex = startIdx;
    updateLightboxImage();
    lightboxEl.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    if (!lightboxEl) return;
    lightboxEl.classList.remove('active');
    document.body.style.overflow = '';
}

function navigateLightbox(dir) {
    currentIndex = (currentIndex + dir + currentImages.length) % currentImages.length;
    updateLightboxImage();
}

function updateLightboxImage() {
    const img = document.getElementById('lb-img');
    const counter = document.getElementById('lb-counter');
    const dots = document.getElementById('lb-dots');

    // Fade transition
    img.style.opacity = '0';
    img.style.transform = 'scale(0.95)';

    setTimeout(() => {
        img.src = currentImages[currentIndex];
        img.onload = () => {
            img.style.opacity = '1';
            img.style.transform = 'scale(1)';
        };
    }, 150);

    counter.textContent = `${currentIndex + 1} / ${currentImages.length}`;

    // Dots
    dots.innerHTML = currentImages.map((_, i) =>
        `<span class="lightbox-dot ${i === currentIndex ? 'active' : ''}" data-idx="${i}"></span>`
    ).join('');

    dots.querySelectorAll('.lightbox-dot').forEach(dot => {
        dot.addEventListener('click', () => {
            currentIndex = parseInt(dot.dataset.idx, 10);
            updateLightboxImage();
        });
    });
}


// ────────────────────────────────────────────────
//  INITIALIZE PAGE
// ────────────────────────────────────────────────
export function initPortfolioPage() {
    // Gallery click handlers
    document.addEventListener('click', (e) => {
        const trigger = e.target.closest('.gallery-trigger');
        if (trigger) {
            e.preventDefault();
            const images = JSON.parse(trigger.dataset.images);
            openLightbox(images, 0);
        }
    });

    // Reveal animations
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

    // Project card hover tilt
    document.querySelectorAll('.project-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const rotateX = ((y - rect.height / 2) / (rect.height / 2)) * -4;
            const rotateY = ((x - rect.width / 2) / (rect.width / 2)) * 4;
            gsap.to(card, { rotateX, rotateY, transformPerspective: 800, duration: 0.3, ease: 'power2.out' });
        });

        card.addEventListener('mouseleave', () => {
            gsap.to(card, { rotateX: 0, rotateY: 0, duration: 0.5, ease: 'power2.out' });
        });
    });
}
