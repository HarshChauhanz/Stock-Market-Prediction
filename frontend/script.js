const form = document.getElementById('prediction-form');
const resultSection = document.getElementById('result-section');
const resDateSpan = document.getElementById('res-date');
const resPriceSpan = document.getElementById('res-price');
let myChart = null;

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const bank = document.getElementById('bank-select').value;
    const period = document.getElementById('prediction-period').value;
    const date = document.getElementById('prediction-date').value;
    const btn = document.getElementById('predict-btn');

    // Show loading state
    btn.textContent = "Predicting...";
    btn.disabled = true;

    try {
        // Call FastAPI Backend
        // NOTE: Ensure port 8000 matches where your uvicorn is running
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bank_name: bank, date: date, period: period })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        alert("Failed to get prediction: " + error.message);
    } finally {
        btn.textContent = "Get Prediction";
        btn.disabled = false;
    }
});

function displayResults(data) {
    resultSection.classList.remove('hidden');
    resDateSpan.textContent = data.target_date;
    resPriceSpan.textContent = "₹" + data.target_prediction.toFixed(2);

    renderChart(data.dates, data.prices, data.bank, data.period);
}

function renderChart(labels, prices, bankName, period) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (myChart) {
        myChart.destroy();
    }

    myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `${bankName} Price Forecast (${period})`,
                data: prices,
                borderColor: '#ffffffff',
                backgroundColor: '#d8b4fe',
                borderWidth: 2,
                fill: true,
                tension: 0.4, // Makes line smooth
                pointRadius: period === 'year' ? 0 : 3 // Hide points for year view to avoid clutter
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (context) => `Price: ₹${context.parsed.y.toFixed(2)}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: { display: true, text: 'Price (INR)' }
                },
                x: {
                    title: { display: true, text: 'Date' },
                    ticks: {
                        maxTicksLimit: period === 'year' ? 12 : (period === 'month' ? 10 : 7)
                    }
                }
            }
        }
    });
}