const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    const isOpen = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', String(!isOpen));
    navLinks.classList.toggle('open', !isOpen);
  });

  navLinks.addEventListener('click', (event) => {
    if (event.target.closest('a')) {
      navToggle.setAttribute('aria-expanded', 'false');
      navLinks.classList.remove('open');
    }
  });
}

const resultTabs = [...document.querySelectorAll('.result-tab')];
const resultPanel = document.querySelector('#result-panel');
const resultImage = document.querySelector('#result-image');
const resultTitle = document.querySelector('#result-title');
const resultDescription = document.querySelector('#result-description');

function activateResult(tab) {
  if (!resultPanel || !resultImage || !resultTitle || !resultDescription) return;

  resultTabs.forEach((candidate) => {
    const isActive = candidate === tab;
    candidate.classList.toggle('active', isActive);
    candidate.setAttribute('aria-selected', String(isActive));
  });

  resultPanel.setAttribute('aria-labelledby', tab.id);
  resultPanel.classList.add('changing');

  const nextImage = new Image();
  nextImage.src = tab.dataset.image;
  nextImage.onload = () => {
    resultImage.src = nextImage.src;
    resultImage.alt = `Direct in-scene comparison under ${tab.dataset.title.toLowerCase()}.`;
    resultTitle.textContent = tab.dataset.title;
    resultDescription.textContent = tab.dataset.description;
    resultPanel.classList.remove('changing');
  };
}

resultTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateResult(tab));
  tab.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === 'ArrowRight' ? 1 : -1;
    const nextIndex = (index + direction + resultTabs.length) % resultTabs.length;
    resultTabs[nextIndex].focus();
    activateResult(resultTabs[nextIndex]);
  });
});

const evidenceCards = [...document.querySelectorAll('.impact-card[data-evidence-target]')];
const evidencePanels = [...document.querySelectorAll('.evidence-panel')];

function activateEvidence(card) {
  evidenceCards.forEach((candidate) => {
    const isActive = candidate === card;
    candidate.classList.toggle('active', isActive);
    candidate.setAttribute('aria-selected', String(isActive));
    candidate.tabIndex = isActive ? 0 : -1;
  });

  evidencePanels.forEach((panel) => {
    const isActive = panel.id === card.dataset.evidenceTarget;
    panel.hidden = !isActive;
    panel.classList.toggle('active', isActive);
  });
}

evidenceCards.forEach((card, index) => {
  card.addEventListener('mouseenter', () => activateEvidence(card));
  card.addEventListener('focus', () => activateEvidence(card));
  card.addEventListener('click', () => activateEvidence(card));
  card.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const forward = event.key === 'ArrowRight' || event.key === 'ArrowDown';
    const nextIndex = (index + (forward ? 1 : -1) + evidenceCards.length) % evidenceCards.length;
    evidenceCards[nextIndex].focus();
  });
});

document.querySelectorAll('[data-copy-target]').forEach((button) => {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;
    const label = button.querySelector('span');

    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      label.textContent = 'Copied';
      button.classList.add('copied');
      window.setTimeout(() => {
        label.textContent = 'Copy';
        button.classList.remove('copied');
      }, 1800);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
      label.textContent = 'Selected';
    }
  });
});

const revealItems = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.08, rootMargin: '0px 0px -24px' },
  );
  revealItems.forEach((item) => revealObserver.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('visible'));
}
