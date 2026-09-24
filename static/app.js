const textarea = document.querySelector('#complaint');
const count = document.querySelector('#count');
const form = document.querySelector('#form');
const button = document.querySelector('#analyse');
const error = document.querySelector('#error');
const result = document.querySelector('#result');
const empty = document.querySelector('#empty');
const samples = {
  'Power cut': 'There has been no electricity in our lane since last night. A transformer is making sparks and immediate help is needed.',
  'Water leak': 'A broken water pipeline near the market has been leaking for three days and wasting a large amount of clean water.',
  'Pothole': 'There is a deep pothole outside the school on the main road. Children are at risk and an accident may happen.'
};

textarea.addEventListener('input', () => count.textContent = textarea.value.length);
document.querySelectorAll('.examples button').forEach(item => item.addEventListener('click', () => {
  textarea.value = samples[item.textContent]; textarea.dispatchEvent(new Event('input')); textarea.focus();
}));

form.addEventListener('submit', async event => {
  event.preventDefault(); error.textContent = ''; button.disabled = true; button.innerHTML = 'Analysing language…';
  try {
    const response = await fetch('/api/analyse', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({complaint:textarea.value})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed.');
    document.querySelector('#category').textContent = data.category;
    document.querySelector('#confidence').textContent = `${data.confidence}%`;
    document.querySelector('#confidence-bar').style.width = `${data.confidence}%`;
    document.querySelector('#urgency').textContent = `${data.urgency_score}%`;
    const priority = document.querySelector('#priority'); priority.textContent = data.priority.toUpperCase(); priority.className = data.priority.toLowerCase();
    document.querySelector('#keywords').innerHTML = data.keywords.map(word => `<i>${word}</i>`).join('');
    document.querySelector('#ranking').innerHTML = data.top_categories.slice(1).map(item => `<div><span>${item.name}</span><i><b style="width:${item.score}%"></b></i><strong>${item.score}%</strong></div>`).join('');
    document.querySelector('#similar-text').textContent = `“${data.similar.text}”`;
    document.querySelector('#similar-meta').textContent = `${data.similar.category} · ${data.similar.similarity}% textual similarity`;
    empty.classList.add('hidden'); result.classList.remove('hidden');
  } catch (err) { error.textContent = err.message; }
  finally { button.disabled = false; button.innerHTML = 'Run NLP analysis <b>→</b>'; }
});

