document.addEventListener('DOMContentLoaded', () => {
    // --- All page logic is now safely inside this listener ---

    // 1. Simple Scroll-In Animation Logic
    const scrollElements = document.querySelectorAll('.scroll-animation');
    const elementInView = (el, dividend = 1) => {
        const elementTop = el.getBoundingClientRect().top;
        return (
            elementTop <= (window.innerHeight || document.documentElement.clientHeight) / dividend
        );
    };
    const displayScrollElement = (element) => {
        element.classList.add('visible');
    };
    const handleScrollAnimation = () => {
        scrollElements.forEach((el) => {
            if (elementInView(el, 1.25)) {
                displayScrollElement(el);
            }
        });
    };
    window.addEventListener('scroll', handleScrollAnimation);
    handleScrollAnimation(); // Initial check on page load

    // 2. Hero Carousel Logic
    const mockup = document.querySelector('.slack-mockup');
    if (mockup) {
        const slides = mockup.querySelectorAll('.slide');
        if (slides.length > 1) {
            let currentSlide = 0;
            let carouselInterval;
            let isPaused = false;

            const advanceSlide = () => {
                if (isPaused) return;
                slides[currentSlide].classList.remove('active');
                currentSlide = (currentSlide + 1) % slides.length;
                slides[currentSlide].classList.add('active');
            };

            const startCarousel = () => {
                if (carouselInterval) clearInterval(carouselInterval);
                carouselInterval = setInterval(advanceSlide, 5000); // 4s display + 1s transition
            };

            const stopCarousel = () => {
                clearInterval(carouselInterval);
            };

            mockup.addEventListener('mouseenter', () => {
                isPaused = true;
                stopCarousel();
            });
            mockup.addEventListener('mouseleave', () => {
                isPaused = false;
                startCarousel();
            });

            mockup.addEventListener('touchstart', (e) => {
                e.preventDefault();
                isPaused = !isPaused;
                if (isPaused) {
                    stopCarousel();
                } else {
                    advanceSlide();
                    startCarousel();
                }
            }, { passive: false });

            startCarousel();
        }
    }

    // 3. FAQ Accordion Logic
    const faqItems = document.querySelectorAll('.faq-item');
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        const answer = item.querySelector('.faq-answer');
        const toggle = item.querySelector('.faq-toggle');

        question.addEventListener('click', () => {
            const isOpen = answer.style.maxHeight && answer.style.maxHeight !== '0px';

            faqItems.forEach(otherItem => {
                if (otherItem !== item) {
                    otherItem.querySelector('.faq-answer').style.maxHeight = '0px';
                    otherItem.querySelector('.faq-toggle').style.transform = 'rotate(0deg)';
                    otherItem.querySelector('.faq-toggle').textContent = '+';
                }
            });

            if (isOpen) {
                answer.style.maxHeight = '0px';
                toggle.style.transform = 'rotate(0deg)';
                toggle.textContent = '+';
            } else {
                answer.style.maxHeight = answer.scrollHeight + 'px';
                toggle.style.transform = 'rotate(45deg)';
                toggle.textContent = '×';
            }
        });
    });

    // 4. Sticky Header CTA Logic
    const headerCta = document.querySelector('#header-cta');
    const headerWaitlistCta = document.querySelector('#header-waitlist-cta');
    const heroSection = document.querySelector('.hero-section');
    const foundersCircleButton = document.querySelector('#founders-circle .cta-button'); // Target the specific button

    const updateHeaderButtonsVisibility = () => {
        const isHeroIntersecting = heroSection ? heroSection.getBoundingClientRect().bottom > 100 : false;
        const isFoundersCircleButtonVisible = foundersCircleButton ? foundersCircleButton.getBoundingClientRect().top < (window.innerHeight || document.documentElement.clientHeight) && foundersCircleButton.getBoundingClientRect().bottom > 0 : false;

        if (!isHeroIntersecting && !isFoundersCircleButtonVisible) {
            headerCta.classList.add('visible');
            if (headerWaitlistCta) {
                headerWaitlistCta.classList.remove('hidden');
            }
        } else {
            headerCta.classList.remove('visible');
            if (headerWaitlistCta) {
                headerWaitlistCta.classList.add('hidden');
            }
        }
    };

    if (headerCta && heroSection && foundersCircleButton) {
        window.addEventListener('scroll', updateHeaderButtonsVisibility);
        updateHeaderButtonsVisibility(); // Initial check on page load
    }

    // 5. Waitlist Form Submission Logic for Netlify Forms with custom validation and toaster
    const waitlistForm = document.querySelector('form[name="waitlist"]');
    if (waitlistForm) {
        const emailInput = waitlistForm.querySelector('input[name="email"]');
        const errorElement = document.getElementById('email-error');
        const toaster = document.getElementById('toaster');

        const validateEmail = (email) => {
            const re = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
            return re.test(String(email).toLowerCase());
        };

        waitlistForm.addEventListener('submit', (e) => {
            e.preventDefault();
            errorElement.textContent = '';
            const email = emailInput.value;

            if (!validateEmail(email)) {
                errorElement.textContent = 'Please enter a valid email address.';
                return;
            }

            const formData = new FormData(waitlistForm);
            fetch('/', {
                method: 'POST',
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: new URLSearchParams(formData).toString()
            }).then(() => {
                emailInput.value = '';
                toaster.style.opacity = '1';
                setTimeout(() => {
                    toaster.style.opacity = '0';
                }, 4000);
            }).catch((error) => {
                errorElement.textContent = 'Something went wrong. Please try again.';
            });
        });
    }

    // 6. Final Purchase Modal Logic with Pre-Save
    const purchaseModal = document.getElementById('purchase-modal');
    const modalContent = document.getElementById('modal-content');
    const founderButtons = document.querySelectorAll('[data-polar-checkout]');
    const proceedButton = document.getElementById('proceed-button');
    const cancelButton = document.getElementById('cancel-button');
    const termsCheckbox = document.getElementById('terms-checkbox');
    const privacyCheckbox = document.getElementById('privacy-checkbox');
    const prePaymentForm = document.getElementById('pre-payment-form');
    let activeFounderButton = null;
    let isSubmitting = false;

    const openModal = (event) => {
        event.stopImmediatePropagation();
        event.preventDefault();
        activeFounderButton = event.currentTarget;
        purchaseModal.style.display = 'flex';
        setTimeout(() => {
            purchaseModal.style.opacity = '1';
            modalContent.style.transform = 'scale(1)';
        }, 10);
    };

    const closeModal = () => {
        purchaseModal.style.opacity = '0';
        modalContent.style.transform = 'scale(0.95)';
        setTimeout(() => {
            purchaseModal.style.display = 'none';
        }, 300);
    };

    const validateInputs = () => {
        const requiredInputs = prePaymentForm.querySelectorAll('[required]');
        let allValid = true;
        requiredInputs.forEach(input => {
            if (!input.value.trim()) {
                allValid = false;
            }
        });
        return allValid && termsCheckbox.checked && privacyCheckbox.checked;
    };

    const updateButtonState = () => {
        if (validateInputs()) {
            proceedButton.classList.remove('opacity-50', 'cursor-not-allowed');
            proceedButton.disabled = false;
        } else {
            proceedButton.classList.add('opacity-50', 'cursor-not-allowed');
            proceedButton.disabled = true;
        }
    };

    founderButtons.forEach(button => {
        button.addEventListener('click', openModal, true);
    });

    cancelButton.addEventListener('click', closeModal);
    document.getElementById('modal-overlay').addEventListener('click', closeModal);
    
    prePaymentForm.addEventListener('input', updateButtonState);
    termsCheckbox.addEventListener('change', updateButtonState);
    privacyCheckbox.addEventListener('change', updateButtonState);

    proceedButton.addEventListener('click', async () => {
        if (isSubmitting || proceedButton.disabled) return;

        isSubmitting = true;
        proceedButton.textContent = 'Processing...';

        const formData = new FormData(prePaymentForm);
        const formProps = Object.fromEntries(formData);

        try {
            const response = await fetch('/.netlify/functions/pre-payment-save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formProps),
            });

            if (!response.ok) {
                throw new Error('Failed to save user data.');
            }

            const { id } = await response.json();

            // Data saved, now proceed to payment
            const checkoutUrl = `${activeFounderButton.href}?client_reference_id=${id}`;
            
            if (window.Polar && typeof window.Polar.checkout === 'function') {
                window.Polar.checkout(checkoutUrl, { theme: 'dark' });
            } else {
                window.open(checkoutUrl, '_blank');
            }
            
            closeModal();

        } catch (error) {
            console.error('Submission error:', error);
            alert('There was an error. Please try again.');
        } finally {
            isSubmitting = false;
            proceedButton.textContent = 'Agree & Continue';
        }
    });

    // Initial check
    updateButtonState();
});
