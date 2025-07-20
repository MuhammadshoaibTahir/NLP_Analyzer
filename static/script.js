// 📜 Ensure vertical scroll to prevent layout shift
window.addEventListener('load', () => {
  document.body.style.overflowY = 'scroll';

  // 🌟 Add premium badge with subtle animation
  if (typeof isPremium !== 'undefined' && isPremium) {
    const header = document.getElementById('header');
    if (header && !document.getElementById('premium-badge')) {
      const badge = document.createElement('span');
      badge.id = 'premium-badge';
      badge.textContent = '★ Premium';
      Object.assign(badge.style, {
        color: '#FFD700',
        marginLeft: '12px',
        fontWeight: '600',
        fontSize: '1rem',
        backgroundColor: '#fff8dc',
        padding: '3px 8px',
        borderRadius: '8px',
        boxShadow: '0 0 6px rgba(0,0,0,0.1)',
        animation: 'fadeIn 1s ease-in-out'
      });
      header.appendChild(badge);
    }
  }
});

// ✍️ Auto-resize <textarea> and enhance UI on user input
document.addEventListener('DOMContentLoaded', () => {
  const textarea = document.querySelector('textarea');
  if (textarea) {
    textarea.style.transition = 'height 0.3s ease';
    textarea.addEventListener('input', () => {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight}px`;
    });
  }

  // 💾 Display and enable the Save button for logged-in users
  if (typeof isLoggedIn !== 'undefined' && isLoggedIn) {
    const saveBtn = document.getElementById('saveBtn');
    const outputArea = document.getElementById('outputArea');

    if (saveBtn && outputArea) {
      saveBtn.style.display = 'inline-flex';
      saveBtn.style.justifyContent = 'center';
      saveBtn.style.alignItems = 'center';
      saveBtn.style.padding = '8px 16px';
      saveBtn.style.marginTop = '10px';
      saveBtn.style.backgroundColor = '#007bff';
      saveBtn.style.color = '#fff';
      saveBtn.style.border = 'none';
      saveBtn.style.borderRadius = '6px';
      saveBtn.style.cursor = 'pointer';
      saveBtn.style.boxShadow = '0 2px 6px rgba(0,0,0,0.15)';
      saveBtn.style.transition = 'all 0.3s ease-in-out';

      saveBtn.addEventListener('mouseover', () => {
        saveBtn.style.backgroundColor = '#0056b3';
      });

      saveBtn.addEventListener('mouseout', () => {
        saveBtn.style.backgroundColor = '#007bff';
      });

      saveBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        const output = outputArea.value.trim();

        if (!output) {
          return alert('⚠️ Please generate or type something to save.');
        }

        try {
          const response = await fetch('/save_output', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ output })
          });

          if (!response.ok) throw new Error('Server responded with error.');
          
          alert('✅ Output saved successfully!');
        } catch (err) {
          console.error('❌ Save error:', err);
          alert('❌ Failed to save output. Try again later.');
        }
      });
    }
  }
});

// 🌈 Optional: Add fadeIn keyframes if you want animation effects via JS
const styleSheet = document.createElement('style');
styleSheet.type = 'text/css';
styleSheet.innerText = `
  @keyframes fadeIn {
    0% { opacity: 0; transform: translateY(-5px); }
    100% { opacity: 1; transform: translateY(0); }
  }
`;
document.head.appendChild(styleSheet);