import { getCurrentUserToken } from './gatekeeper.js?v=79.6';

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('reviewForm');
    const stars = document.querySelectorAll('.star');
    const resultDiv = document.getElementById('reviewResult');
    let currentRating = 5;

    function renderStars() {
        stars.forEach(s => {
            const v = parseInt(s.dataset.value, 10);
            s.textContent = v <= currentRating ? '★' : '☆';
        });
    }

    stars.forEach(s => {
        s.addEventListener('click', () => {
            currentRating = parseInt(s.dataset.value, 10);
            renderStars();
        });
        s.addEventListener('mouseover', () => {
            const v = parseInt(s.dataset.value, 10);
            stars.forEach(x => x.textContent = parseInt(x.dataset.value,10) <= v ? '★' : '☆');
        });
        s.addEventListener('mouseout', renderStars);
    });
    renderStars();

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        resultDiv.textContent = '';
        const name = document.getElementById('reviewName').value.trim();
        const email = document.getElementById('reviewEmail').value.trim();
        const message = document.getElementById('reviewMessage').value.trim();
        const payload = { name, email, rating: currentRating, message };
        if (!name || !email) {
            resultDiv.style.color = 'var(--danger-red)';
            resultDiv.textContent = 'Please provide name and email.';
            return;
        }
        try {
            const token = await getCurrentUserToken();
            const headers = { 'Content-Type': 'application/json' };
            if (token) headers['Authorization'] = `Bearer ${token}`;
            const res = await fetch(window.location.origin + '/reviews', {
                method: 'POST',
                headers,
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (res.ok) {
                resultDiv.style.color = 'var(--success-green)';
                resultDiv.textContent = 'Thanks — your review was submitted.';
                form.reset();
                currentRating = 5; renderStars();
            } else {
                resultDiv.style.color = 'var(--danger-red)';
                resultDiv.textContent = data.detail || 'Failed to submit review.';
            }
        } catch (err) {
            console.error('Review submit error:', err);
            resultDiv.style.color = 'var(--danger-red)';
            resultDiv.textContent = 'Failed to submit review. Please try again later.';
        }
    });
});
