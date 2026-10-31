const API_BASE_URL = "http://127.0.0.1:8000/api/v1";
let tvChart = null;
let lineSeries = null;
let predictionSeries = null;
let upperBandSeries = null;
let lowerBandSeries = null;
let currentDataPayload = null;

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    runAnalysis();
});

function initChart() {
    const chartContainer = document.getElementById('tvChart');
    tvChart = LightweightCharts.createChart(chartContainer, {
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#94A3B8',
            fontFamily: "'JetBrains Mono', monospace",
        },
        grid: {
            vertLines: { color: 'rgba(33, 43, 54, 0.4)' },
            horzLines: { color: 'rgba(33, 43, 54, 0.4)' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: '#00F090', width: 1, style: LightweightCharts.LineStyle.Dashed },
            horzLine: { color: '#00F090', width: 1, style: LightweightCharts.LineStyle.Dashed },
        },
        rightPriceScale: {
            borderColor: '#212B36',
        },
        timeScale: {
            borderColor: '#212B36',
            timeVisible: true,
        },
    });

    // Upper & Lower Confidence Volatility Bands (as lines)
    upperBandSeries = tvChart.addLineSeries({
        color: 'rgba(0, 240, 144, 0.4)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        crosshairMarkerVisible: false,
    });
    lowerBandSeries = tvChart.addLineSeries({
        color: 'rgba(255, 59, 48, 0.4)',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        crosshairMarkerVisible: false,
    });

    // Historical Price Series (Partition 1: Current & Prev Status)
    lineSeries = tvChart.addAreaSeries({
        lineColor: '#3B82F6',
        topColor: 'rgba(59, 130, 246, 0.4)',
        bottomColor: 'rgba(59, 130, 246, 0.0)',
        lineWidth: 2,
        crosshairMarkerVisible: true,
    });

    // Bi-LSTM Forecast Series (Partition 2: Predicted Status)
    predictionSeries = tvChart.addAreaSeries({
        lineColor: '#00F090',
        topColor: 'rgba(0, 240, 144, 0.5)',
        bottomColor: 'rgba(0, 240, 144, 0.0)',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        crosshairMarkerVisible: true,
    });

    // Interactive Hover Tooltip
    tvChart.subscribeCrosshairMove(param => {
        const tooltip = document.getElementById('chartTooltip');
        if (!param.time || param.point === undefined || param.point.x < 0 || param.point.y < 0) {
            tooltip.classList.add('hidden');
            return;
        }

        const price = param.seriesPrices.get(lineSeries) || param.seriesPrices.get(predictionSeries);
        if (!price) {
            tooltip.classList.add('hidden');
            return;
        }

        const dateStr = param.time;
        const sentiment = currentDataPayload ? currentDataPayload.metrics.sentiment_score : 0;
        const targetPrice = currentDataPayload ? currentDataPayload.metrics.predicted_price : 0;

        tooltip.innerHTML = `
            <div><strong>Date:</strong> ${dateStr}</div>
            <div><strong>Tick Price:</strong> ${formatCurrency(price)}</div>
            <div><strong>FinBERT Score:</strong> <span style="color: ${sentiment >= 0 ? '#00F090' : '#FF3B30'}">${sentiment}</span></div>
            <div><strong>Target Target:</strong> ${formatCurrency(targetPrice)}</div>
        `;
        tooltip.classList.remove('hidden');
    });

    // Responsive Chart Resize
    new ResizeObserver(entries => {
        if (entries.length === 0 || entries[0].target !== chartContainer) { return; }
        const newRect = entries[0].contentRect;
        tvChart.applyOptions({ height: newRect.height, width: newRect.width });
    }).observe(chartContainer);
}

async function runAnalysis() {
    const ticker = document.getElementById("tickerSelect").value;
    // Find active interval
    const activeIntervalBtn = document.querySelector('.interval-btn.active');
    const interval = activeIntervalBtn ? activeIntervalBtn.innerText : '24H';

    const loader = document.getElementById("loader");
    loader.classList.remove("hidden");
    const startTime = performance.now();

    try {
        const response = await fetch(`${API_BASE_URL}/analyze?ticker=${ticker}&interval=${interval}`);
        if (!response.ok) throw new Error(`Server status: ${response.status}`);
        
        const data = await response.json();
        currentDataPayload = data;
        
        const endTime = performance.now();
        const latency = (endTime - startTime).toFixed(0);
        document.getElementById("apiLatency").innerText = `Latency: ${latency} ms`;
        document.getElementById("inferenceSpeed").innerText = `Inference: ~${Math.max(8, latency - 15)}ms`;
        
        updateDashboard(data);
    } catch (error) {
        console.error(error);
        alert(`Failed to fetch analysis: ${error.message}`);
    } finally {
        loader.classList.add("hidden");
    }
}

function updateDashboard(data) {
    const metrics = data.metrics;
    
    // Core Metrics
    document.getElementById("currentPrice").innerText = formatCurrency(metrics.current_price);
    document.getElementById("predictedPrice").innerText = formatCurrency(metrics.predicted_price);
    
    const changeElem = document.getElementById("expectedChange");
    const changeSign = metrics.expected_change_pct >= 0 ? "+" : "";
    changeElem.innerText = `${changeSign}${metrics.expected_change_pct}%`;
    changeElem.className = `value mono ${metrics.expected_change_pct >= 0 ? 'bullish-text' : 'bearish-text'}`;

    // Direction Badge
    const badge = document.getElementById("directionBadge");
    badge.innerText = metrics.direction;
    badge.className = `badge ${metrics.direction.toLowerCase()}`;

    // FinBERT Sentiment Breakdown
    const sScore = metrics.sentiment_score;
    document.getElementById("sentimentScore").innerText = sScore.toFixed(3);
    const polarityLabel = document.getElementById("sentimentPolarity");
    const barFill = document.getElementById("sentimentBarFill");
    
    // Map -1.0..+1.0 to 0%..100%
    const fillPct = Math.min(100, Math.max(0, ((sScore + 1) / 2) * 100));
    barFill.style.width = `${fillPct}%`;

    if (sScore > 0.05) {
        polarityLabel.innerText = "BULLISH";
        polarityLabel.className = "polarity-badge bullish-text";
    } else if (sScore < -0.05) {
        polarityLabel.innerText = "BEARISH";
        polarityLabel.className = "polarity-badge bearish-text";
    } else {
        polarityLabel.innerText = "NEUTRAL";
        polarityLabel.className = "polarity-badge";
    }

    // Drivers
    const fgVal = metrics.fear_greed_score;
    document.getElementById("fearGreedScore").innerText = fgVal.toFixed(2);
    const fgStatus = document.getElementById("fearGreedStatus");
    if (fgVal > 0.2) fgStatus.innerText = "Greed";
    else if (fgVal < -0.2) fgStatus.innerText = "Fear";
    else fgStatus.innerText = "Neutral";

    document.getElementById("gasMetric").innerText = metrics.gas_metric.toFixed(2);

    // AI Brief
    document.getElementById("analystReport").innerText = data.analyst_report;

    // Headlines Feed
    const newsList = document.getElementById("newsList");
    newsList.innerHTML = "";
    if (data.headlines && data.headlines.length > 0) {
        data.headlines.forEach(headline => {
            const li = document.createElement("li");
            li.innerHTML = `• ${headline}`;
            newsList.appendChild(li);
        });
    } else {
        newsList.innerHTML = `<li class="empty-state">No news feed parsed.</li>`;
    }

    // 24H High/Low stats from price series
    if (data.chart_data && data.chart_data.prices) {
        const validPrices = data.chart_data.prices.slice(0, -1); // exclude predicted
        if (validPrices.length > 0) {
            const high = Math.max(...validPrices);
            const low = Math.min(...validPrices);
            document.getElementById("statHigh").innerText = formatCurrency(high);
            document.getElementById("statLow").innerText = formatCurrency(low);
            const vol = (((high - low) / low) * 100).toFixed(2);
            document.getElementById("statVolatility").innerText = `±${vol}%`;
        }
    }

    // Header Marquee
    updateTickerMarquee(data.ticker, metrics);

    // Chart Update
    updateChart(data.chart_data, metrics.current_price, metrics.predicted_price);
}

function updateTickerMarquee(ticker, metrics) {
    const marquee = document.getElementById("liveTickerInfo");
    const changeColor = metrics.expected_change_pct >= 0 ? 'var(--accent-bull)' : 'var(--accent-bear)';
    marquee.innerHTML = `
        <span class="ticker-item"><strong>${ticker}</strong></span>
        <span class="ticker-item">Live: ${formatCurrency(metrics.current_price)}</span>
        <span class="ticker-item">Bi-LSTM Target: ${formatCurrency(metrics.predicted_price)} <span style="color:${changeColor}">(${metrics.expected_change_pct}%)</span></span>
        <span class="ticker-item">Fear & Greed: ${metrics.fear_greed_score}</span>
        <span class="ticker-item">Gas: ${metrics.gas_metric}</span>
    `;
}

function updateChart(chartData, currentPrice, predictedPrice) {
    const histData = [];
    
    for (let i = 0; i < chartData.labels.length; i++) {
        const lbl = chartData.labels[i];
        const prc = chartData.prices[i];
        if (lbl !== "Predicted") {
            histData.push({ time: lbl, value: prc });
        }
    }
    
    lineSeries.setData(histData);

    if (histData.length > 0) {
        const lastHist = histData[histData.length - 1];
        
        // Add marker to clearly indicate current state vs predicted area
        lineSeries.setMarkers([
            {
                time: lastHist.time,
                position: 'aboveBar',
                color: '#3B82F6',
                shape: 'arrowDown',
                text: 'Current Status',
            }
        ]);

        const lastDate = new Date(lastHist.time);
        lastDate.setDate(lastDate.getDate() + 1);
        const futureTime = lastDate.toISOString().split('T')[0];
        
        // Prediction area partition
        predictionSeries.setData([
            lastHist,
            { time: futureTime, value: predictedPrice }
        ]);

        predictionSeries.setMarkers([
            {
                time: futureTime,
                position: 'aboveBar',
                color: '#00F090',
                shape: 'arrowDown',
                text: 'Predicted Target',
            }
        ]);

        // Volatility Band Bounds (e.g. ±2% around target prediction)
        const spread = predictedPrice * 0.02;
        upperBandSeries.setData([
            { time: lastHist.time, value: lastHist.value },
            { time: futureTime, value: predictedPrice + spread }
        ]);
        lowerBandSeries.setData([
            { time: lastHist.time, value: lastHist.value },
            { time: futureTime, value: predictedPrice - spread }
        ]);

        tvChart.timeScale().fitContent();
    }
}

function changeInterval(label, btnElement) {
    document.querySelectorAll('.interval-btn').forEach(btn => btn.classList.remove('active'));
    btnElement.classList.add('active');
    // Triggers fresh analysis reload
    runAnalysis();
}

function formatCurrency(val) {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
}