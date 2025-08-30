document.getElementById('predict-form').addEventListener('submit', async e => {
    e.preventDefault();
    const form = e.target;
    const data = {
        location: form.location.value,
        time: form.time.value,
        day: form.day.value,
        season: form.season.value
    };
    const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(data)
    });
    const result = await res.json();
    document.getElementById('prediction-result').textContent = JSON.stringify(result, null, 2);
});

async function fetchTrends() {
    const res = await fetch('/api/trends');
    const data = await res.json();
    document.getElementById('api-result').textContent = JSON.stringify(data, null, 2);
}

async function fetchHotspots() {
    const res = await fetch('/api/hotspots');
    const data = await res.json();
    document.getElementById('api-result').textContent = JSON.stringify(data, null, 2);
}

async function fetchStatistics() {
    const res = await fetch('/api/statistics');
    const data = await res.json();
    document.getElementById('api-result').textContent = JSON.stringify(data, null, 2);
}

async function fetchHistory() {
    const res = await fetch('/api/history');
    const data = await res.json();
    document.getElementById('api-result').textContent = JSON.stringify(data, null, 2);
}
