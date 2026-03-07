/**
 * ContactForm — Floating labels, glow focus, smooth validation.
 */

export function initContactForm() {
    const form = document.getElementById('contact-form');
    if (!form) return;

    form.addEventListener('submit', (e) => {
        e.preventDefault();

        const name = document.getElementById('contact-name');
        const email = document.getElementById('contact-email');
        const message = document.getElementById('contact-message');

        let valid = true;

        [name, email, message].forEach((field) => {
            if (!field) return;
            if (!field.value.trim()) {
                valid = false;
                shakeField(field);
            }
        });

        // Email regex
        if (email && email.value.trim() && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value.trim())) {
            valid = false;
            shakeField(email);
        }

        if (valid) {
            const submitBtn = document.getElementById('contact-submit');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="bx bx-check text-lg"></i><span>Sent!</span>';
                submitBtn.classList.add('!bg-emerald-500/20', '!border-emerald-500/40', '!text-emerald-400');
                setTimeout(() => {
                    submitBtn.innerHTML = '<i class="bx bx-send"></i><span>Send Message</span>';
                    submitBtn.classList.remove('!bg-emerald-500/20', '!border-emerald-500/40', '!text-emerald-400');
                    form.reset();
                }, 2500);
            }
        }
    });

    function shakeField(field) {
        field.classList.add('!border-red-500/50');
        field.style.animation = 'none';
        field.offsetHeight; // force reflow
        field.style.animation = 'shake 0.4s ease';
        setTimeout(() => {
            field.classList.remove('!border-red-500/50');
        }, 2000);
    }

    // Add shake keyframes once
    if (!document.getElementById('shake-style')) {
        const style = document.createElement('style');
        style.id = 'shake-style';
        style.textContent = `
      @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-6px); }
        50% { transform: translateX(6px); }
        75% { transform: translateX(-4px); }
      }
    `;
        document.head.appendChild(style);
    }
}
