/**
 * Navbar — Glassmorphism sticky nav with mobile menu.
 */

export function initNavbar() {
    const toggle = document.getElementById('mobile-toggle');
    const menu = document.getElementById('mobile-menu');
    const icon = toggle?.querySelector('i');

    if (!toggle || !menu) return;

    let isOpen = false;

    toggle.addEventListener('click', () => {
        isOpen = !isOpen;
        menu.classList.toggle('hidden', !isOpen);
        if (icon) {
            icon.className = isOpen ? 'bx bx-x text-xl' : 'bx bx-menu text-xl';
        }
    });

    // Close on mobile link click
    menu.querySelectorAll('.mobile-link').forEach((link) => {
        link.addEventListener('click', () => {
            isOpen = false;
            menu.classList.add('hidden');
            if (icon) icon.className = 'bx bx-menu text-xl';
        });
    });

    // ─── Active link highlight ───
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');

    function highlightNav() {
        const scrollY = window.scrollY + 120;
        sections.forEach((section) => {
            const top = section.offsetTop;
            const height = section.offsetHeight;
            const id = section.getAttribute('id');

            if (scrollY >= top && scrollY < top + height) {
                navLinks.forEach((link) => {
                    link.classList.remove('!text-neon-cyan', 'bg-neon-cyan/5');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('!text-neon-cyan', 'bg-neon-cyan/5');
                    }
                });
            }
        });
    }

    window.addEventListener('scroll', highlightNav, { passive: true });
    highlightNav();
}
