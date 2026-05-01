"""Simple HTML dashboard for monitoring metrics."""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Credit Risk Agent - Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        .cards { display: flex; gap: 20px; flex-wrap: wrap; }
        .card {
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 200px;
        }
        .card h3 { margin: 0 0 10px 0; color: #666; font-size: 14px; }
        .card .value { font-size: 32px; font-weight: bold; color: #333; }
        .risk-low { color: #27ae60; }
        .risk-medium { color: #f39c12; }
        .risk-high { color: #e74c3c; }
        #status { margin-top: 10px; color: #999; font-size: 12px; }
    </style>
</head>
<body>
    <h1>Credit Risk Agent - Dashboard</h1>
    <div class="cards" id="cards"></div>
    <p id="status">Loading...</p>

    <script>
        async function loadMetrics() {
            try {
                const res = await fetch('/metrics');
                const data = await res.json();

                const cards = document.getElementById('cards');
                cards.innerHTML = `
                    <div class="card">
                        <h3>Uptime</h3>
                        <div class="value">${Math.floor(data.uptime_seconds / 60)}m</div>
                    </div>
                    <div class="card">
                        <h3>Total Predictions</h3>
                        <div class="value">${data.total_predictions}</div>
                    </div>
                    <div class="card">
                        <h3>Avg Latency</h3>
                        <div class="value">${data.avg_latency_ms}ms</div>
                    </div>
                    <div class="card">
                        <h3>Low Risk</h3>
                        <div class="value risk-low">${data.risk_distribution.LOW || 0}</div>
                    </div>
                    <div class="card">
                        <h3>Medium Risk</h3>
                        <div class="value risk-medium">${data.risk_distribution.MEDIUM || 0}</div>
                    </div>
                    <div class="card">
                        <h3>High Risk</h3>
                        <div class="value risk-high">${data.risk_distribution.HIGH || 0}</div>
                    </div>
                `;
                document.getElementById('status').textContent = 'Updated: ' + new Date().toLocaleTimeString();
            } catch (e) {
                document.getElementById('status').textContent = 'Error loading metrics';
            }
        }

        loadMetrics();
        setInterval(loadMetrics, 5000);
    </script>
</body>
</html>
"""
