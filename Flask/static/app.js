document.addEventListener("DOMContentLoaded", () => {
    const revealElements = document.querySelectorAll(".reveal");
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.15 }
    );

    revealElements.forEach((element, index) => {
        const delay = element.dataset.delay
            ? parseInt(element.dataset.delay, 10)
            : index * 40;
        element.style.setProperty("--delay", `${delay}ms`);
        observer.observe(element);
    });
});
