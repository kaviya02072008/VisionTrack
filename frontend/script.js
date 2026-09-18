const API_BASE_URL = 'http://127.0.0.1:8000';

// ── Chart instances ──────────────────────────────────────────────────────────
let occupancyChart = null;
let dwellChart = null;

// ── Tracked UI state ─────────────────────────────────────────────────────────
let currentIsDemo = true;
let heatmapVisible = true;

// ═════════════════════════════════════════════════════════════════════════════
// UTILITY
// ═════════════════════════════════════════════════════════════════════════════

function showError(msg) {
    const banner = document.getElementById('error-banner');
    document.getElementById('error-text').textContent = msg;
    banner.classList.remove('hidden');
}

function setStatus(text, colorClass) {
    const el = document.getElementById('upload-status');
    el.textContent = text;
    el.className = `text-sm ${colorClass}`;
}

// ═════════════════════════════════════════════════════════════════════════════
// CHART INITIALISATION
// ═════════════════════════════════════════════════════════════════════════════

function initCharts() {
    // Occupancy line chart
    const octx = document.getElementById('occupancyChart').getContext('2d');
    occupancyChart = new Chart(octx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Occupancy',
                data: [],
                borderColor: '#a855f7',
                backgroundColor: 'rgba(168,85,247,0.08)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#64748b', maxTicksLimit: 8 },
                    grid: { color: 'rgba(71,85,105,0.2)' }
                },
                y: {
                    beginAtZero: true,
                    ticks: { color: '#64748b', stepSize: 1 },
                    grid: { color: 'rgba(71,85,105,0.2)' }
                }
            }
        }
    });

    // Dwell bar chart
    const dctx = document.getElementById('dwellChart').getContext('2d');
    dwellChart = new Chart(dctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Avg Dwell (s)',
                data: [],
                backgroundColor: [
                    'rgba(251,191,36,0.6)',
                    'rgba(168,85,247,0.6)',
                    'rgba(34,211,238,0.6)',
                    'rgba(52,211,153,0.6)',
                ],
                borderColor: [
                    'rgba(251,191,36,1)',
                    'rgba(168,85,247,1)',
                    'rgba(34,211,238,1)',
                    'rgba(52,211,153,1)',
                ],
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { display: false } },
                y: { beginAtZero: true, ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(71,85,105,0.2)' } }
            }
        }
    });
}

// ═════════════════════════════════════════════════════════════════════════════
// STATS POLLING
// ═════════════════════════════════════════════════════════════════════════════

async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE_URL}/stats`);
        if (!res.ok) { showError('Backend unavailable. Check that the server is running.'); return; }
        const data = await res.json();
        updateDashboard(data);
    } catch (e) {
        showError('Cannot reach backend (http://127.0.0.1:8000). Is the server running?');
    }
}

function updateDashboard(data) {
    if (!data) return;

    currentIsDemo = !!data.is_demo;

    // ── Header / mode badge ──────────────────────────────────────────────────
    const modeBadge = document.getElementById('mode-badge');
    const systemStatus = document.getElementById('system-status');
    const liveDot = document.getElementById('live-dot');
    const demoBadge = document.getElementById('demo-badge');
    const exportNotice = document.getElementById('export-notice');

    if (currentIsDemo) {
        modeBadge.textContent = 'DEMO MODE — SIMULATED DATA';
        modeBadge.className = 'mode-badge-demo px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider';
        systemStatus.textContent = 'Demo Mode Active';
        liveDot.children[0].className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75';
        liveDot.children[1].className = 'relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-500';
        demoBadge.classList.remove('hidden');
        exportNotice.classList.remove('hidden');
    } else {
        modeBadge.textContent = 'REAL VIDEO ANALYSIS';
        modeBadge.className = 'mode-badge-real px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider';
        systemStatus.textContent = 'Real Video — System Active';
        liveDot.children[0].className = 'animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75';
        liveDot.children[1].className = 'relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500';
        demoBadge.classList.add('hidden');
        exportNotice.classList.add('hidden');
    }

    // ── KPIs ─────────────────────────────────────────────────────────────────
    document.getElementById('stat-current').textContent = data.current_occupancy ?? 0;
    document.getElementById('stat-peak').textContent = data.peak_occupancy ?? 0;
    document.getElementById('stat-avg').textContent = data.average_occupancy ?? 0.0;
    document.getElementById('stat-unique').textContent = data.unique_people ?? 0;

    const avgDwell = data.average_dwell_global ?? 0.0;
    document.getElementById('stat-dwell').textContent = `${avgDwell}s`;

    // ── Occupancy chart ───────────────────────────────────────────────────────
    if (data.history && occupancyChart) {
        occupancyChart.data.labels = data.history.map(h => h.frame);
        occupancyChart.data.datasets[0].data = data.history.map(h => h.occupancy);
        occupancyChart.update('none');
    }

    // ── Dwell table + chart ──────────────────────────────────────────────────
    if (data.dwell_stats) {
        updateDwellTable(data.dwell_stats);
        updateDwellChart(data.dwell_stats);
    }

    // ── Traffic table ────────────────────────────────────────────────────────
    if (data.traffic_stats) {
        updateTrafficTable(data.traffic_stats);
    }

    // ── Movement patterns ────────────────────────────────────────────────────
    if (data.zone_transitions && data.transition_matrix) {
        updateMovementPatterns(data.zone_transitions, data.transition_matrix, data.zone_names || []);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// DWELL TABLE & CHART
// ═════════════════════════════════════════════════════════════════════════════

function updateDwellTable(dwellStats) {
    const tbody = document.getElementById('dwell-body');
    if (!dwellStats.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="py-2 text-slate-500 text-xs italic">No dwell data yet.</td></tr>';
        return;
    }
    tbody.innerHTML = dwellStats.map(d => `
        <tr class="border-b border-slate-800/50">
            <td class="py-1.5 pr-2 font-medium">${d.zone}</td>
            <td class="py-1.5 pr-2">${d.visits}</td>
            <td class="py-1.5 pr-2 text-amber-400">${d.average_dwell_seconds}</td>
            <td class="py-1.5 pr-2 text-slate-400">${d.minimum_dwell_seconds}</td>
            <td class="py-1.5 text-slate-400">${d.maximum_dwell_seconds}</td>
        </tr>
    `).join('');
}

function updateDwellChart(dwellStats) {
    if (!dwellChart) return;
    dwellChart.data.labels = dwellStats.map(d => d.zone);
    dwellChart.data.datasets[0].data = dwellStats.map(d => d.average_dwell_seconds);
    dwellChart.update('none');
}

// ═════════════════════════════════════════════════════════════════════════════
// TRAFFIC TABLE
// ═════════════════════════════════════════════════════════════════════════════

function trafficBadge(level) {
    const map = {
        HIGH:   'badge-high',
        MEDIUM: 'badge-medium',
        LOW:    'badge-low',
    };
    return `<span class="${map[level] || 'badge-low'}">${level}</span>`;
}

function updateTrafficTable(trafficStats) {
    const tbody = document.getElementById('traffic-body');
    const highlights = document.getElementById('traffic-highlights');

    if (!trafficStats.length) {
        tbody.innerHTML = '<tr><td colspan="4" class="py-2 text-slate-500 text-xs italic">No traffic data yet.</td></tr>';
        highlights.innerHTML = '';
        return;
    }

    tbody.innerHTML = trafficStats.map(t => `
        <tr class="border-b border-slate-800/50">
            <td class="py-1.5 pr-2 font-medium">${t.zone}</td>
            <td class="py-1.5 pr-2">${t.visits}</td>
            <td class="py-1.5 pr-2 text-amber-400">${t.average_dwell_seconds}s</td>
            <td class="py-1.5">${trafficBadge(t.traffic_level)}</td>
        </tr>
    `).join('');

    // Highlight HIGH-traffic and LOW-traffic zones
    const high = trafficStats.filter(t => t.traffic_level === 'HIGH').map(t => t.zone);
    const low  = trafficStats.filter(t => t.traffic_level === 'LOW').map(t => t.zone);

    highlights.innerHTML = '';
    if (high.length) {
        highlights.innerHTML += `
            <div class="bg-red-900/30 border border-red-700/40 rounded-lg px-3 py-2 text-xs">
                <span class="text-red-400 font-semibold uppercase tracking-wider">High-Traffic Areas</span>
                <p class="text-slate-300 mt-0.5">${high.join(', ')}</p>
            </div>`;
    }
    if (low.length) {
        highlights.innerHTML += `
            <div class="bg-slate-800/60 border border-slate-700/40 rounded-lg px-3 py-2 text-xs mt-2">
                <span class="text-slate-400 font-semibold uppercase tracking-wider">Low-Traffic Areas</span>
                <p class="text-slate-300 mt-0.5">${low.join(', ')}</p>
            </div>`;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// MOVEMENT PATTERNS & TRANSITION MATRIX
// ═════════════════════════════════════════════════════════════════════════════

function updateMovementPatterns(transitions, matrix, zoneNames) {
    // Common path: find the longest sequence among all track transitions
    let longestPath = [];
    for (const zones of Object.values(transitions)) {
        if (Array.isArray(zones) && zones.length > longestPath.length) {
            longestPath = zones;
        }
    }

    const pathEl = document.getElementById('common-path');
    if (longestPath.length > 1) {
        // Deduplicate consecutive duplicates for readability
        const deduped = longestPath.filter((z, i) => i === 0 || z !== longestPath[i - 1]);
        pathEl.textContent = deduped.join(' → ');
        pathEl.classList.remove('italic');
    } else {
        pathEl.textContent = 'Collecting data…';
        pathEl.classList.add('italic');
    }

    // Transition matrix
    const tbody = document.getElementById('matrix-body');
    if (!zoneNames.length) { tbody.innerHTML = ''; return; }

    // Header row
    let headerRow = '<tr><th class="text-slate-600 text-xs pb-1 pr-1 text-left">↓ From \\ To →</th>';
    zoneNames.forEach(z => {
        headerRow += `<th class="text-slate-500 text-xs pb-1 pr-1 text-center whitespace-nowrap">${z.split(' ')[0]}</th>`;
    });
    headerRow += '</tr>';

    // Data rows
    let dataRows = '';
    zoneNames.forEach(fromZ => {
        dataRows += `<tr><td class="text-slate-400 text-xs py-0.5 pr-2 font-medium whitespace-nowrap">${fromZ}</td>`;
        zoneNames.forEach(toZ => {
            const count = (matrix[fromZ] && matrix[fromZ][toZ] != null) ? matrix[fromZ][toZ] : 0;
            const cls = count > 0 ? 'text-cyan-400 font-medium' : 'text-slate-700';
            dataRows += `<td class="text-xs py-0.5 pr-1 text-center ${cls}">${count}</td>`;
        });
        dataRows += '</tr>';
    });

    tbody.innerHTML = headerRow + dataRows;
}

// ═════════════════════════════════════════════════════════════════════════════
// VIDEO INFO
// ═════════════════════════════════════════════════════════════════════════════

async function fetchVideoInfo() {
    const container = document.getElementById('video-info-container');
    try {
        const res = await fetch(`${API_BASE_URL}/video_info`);
        if (!res.ok) {
            container.innerHTML = '<p class="text-amber-400 text-xs italic">No video selected — running Demo Mode</p>';
            return;
        }
        const d = await res.json();
        container.innerHTML = `
            <div class="grid grid-cols-2 gap-y-1.5 text-xs">
                <span class="text-slate-500">Filename</span><span class="text-slate-200 font-medium truncate">${d.filename}</span>
                <span class="text-slate-500">Resolution</span><span class="text-slate-200 font-medium">${d.width} × ${d.height}</span>
                <span class="text-slate-500">FPS</span><span class="text-slate-200 font-medium">${d.fps?.toFixed(2)}</span>
                <span class="text-slate-500">Duration</span><span class="text-slate-200 font-medium">${d.duration_seconds?.toFixed(1)}s</span>
                <span class="text-slate-500">Frames</span><span class="text-slate-200 font-medium">${d.frame_count}</span>
            </div>`;
    } catch (e) {
        container.innerHTML = '<p class="text-red-400 text-xs">Could not fetch video info.</p>';
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// DATASET VIDEO LIST
// ═════════════════════════════════════════════════════════════════════════════

async function loadDatasetVideos() {
    try {
        const res = await fetch(`${API_BASE_URL}/dataset_videos`);
        const data = await res.json();
        const select = document.getElementById('dataset-select');
        select.innerHTML = '<option value="">— Select a video —</option>';

        if (data.videos?.length) {
            data.videos.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.path;
                opt.textContent = v.filename;
                select.appendChild(opt);
            });
        } else {
            select.innerHTML = '<option value="">No videos found in dataset/</option>';
        }
    } catch (e) {
        document.getElementById('dataset-select').innerHTML = '<option value="">Could not load videos</option>';
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// FEED REFRESH
// ═════════════════════════════════════════════════════════════════════════════

function refreshFeeds() {
    const t = Date.now();
    document.getElementById('video-feed').src = `${API_BASE_URL}/video_feed?t=${t}`;
    document.getElementById('heatmap-feed').src = `${API_BASE_URL}/heatmap_feed?t=${t}`;
}

// ═════════════════════════════════════════════════════════════════════════════
// EVENT LISTENERS
// ═════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    loadDatasetVideos();
    fetchVideoInfo();
    setInterval(fetchStats, 500);

    const statusEl = document.getElementById('upload-status');

    // ── Upload ────────────────────────────────────────────────────────────────
    document.getElementById('btn-upload').addEventListener('click', async () => {
        const fileInput = document.getElementById('video-upload-file');
        if (!fileInput.files.length) {
            setStatus('Please select a file first.', 'text-red-400');
            return;
        }
        const ext = fileInput.files[0].name.split('.').pop().toLowerCase();
        if (!['mp4', 'avi', 'mov'].includes(ext)) {
            setStatus(`Unsupported format ".${ext}". Use .mp4, .avi, or .mov.`, 'text-red-400');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        setStatus('Uploading…', 'text-blue-400');

        try {
            const res = await fetch(`${API_BASE_URL}/upload_video`, { method: 'POST', body: formData });
            const data = await res.json();
            if (data.success) {
                setStatus(`✓ Uploaded: ${data.filename}`, 'text-emerald-400');
                fileInput.value = '';
                loadDatasetVideos();
            } else {
                setStatus(`✗ ${data.error}`, 'text-red-400');
            }
        } catch (e) {
            setStatus('Upload failed — network error.', 'text-red-400');
        }
    });

    // ── Load dataset video ────────────────────────────────────────────────────
    document.getElementById('btn-load').addEventListener('click', async () => {
        const path = document.getElementById('dataset-select').value;
        if (!path) { setStatus('Select a video from the dropdown first.', 'text-amber-400'); return; }
        setStatus('Loading…', 'text-blue-400');

        try {
            const res = await fetch(`${API_BASE_URL}/select_video`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'REAL', path }),
            });
            const data = await res.json();
            if (data.success) {
                setStatus('✓ Video loaded — ready for analysis.', 'text-emerald-400');
                fetchVideoInfo();
                refreshFeeds();
            } else {
                setStatus(`✗ ${data.error}`, 'text-red-400');
            }
        } catch (e) {
            setStatus('Failed to load video.', 'text-red-400');
        }
    });

    // ── Demo mode ─────────────────────────────────────────────────────────────
    document.getElementById('btn-demo').addEventListener('click', async () => {
        setStatus('Switching to Demo Mode…', 'text-amber-400');
        try {
            const res = await fetch(`${API_BASE_URL}/select_video`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: 'DEMO', path: '' }),
            });
            const data = await res.json();
            if (data.success) {
                setStatus('Demo Mode active.', 'text-amber-400');
                fetchVideoInfo();
                refreshFeeds();
            }
        } catch (e) {
            setStatus('Failed to switch to Demo Mode.', 'text-red-400');
        }
    });

    // ── Start analysis ────────────────────────────────────────────────────────
    document.getElementById('btn-start').addEventListener('click', async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/control/start`, { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                setStatus(`▶ Analysis running (${data.mode} mode).`, 'text-emerald-400');
                refreshFeeds();
            }
        } catch (e) {
            setStatus('Cannot reach backend.', 'text-red-400');
        }
    });

    // ── Stop analysis ─────────────────────────────────────────────────────────
    document.getElementById('btn-stop').addEventListener('click', async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/control/stop`, { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                setStatus('⏸ Analysis paused.', 'text-amber-400');
            }
        } catch (e) {
            setStatus('Cannot reach backend.', 'text-red-400');
        }
    });

    // ── Reset analytics ───────────────────────────────────────────────────────
    document.getElementById('btn-reset').addEventListener('click', async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/control/reset`, { method: 'POST' });
            const data = await res.json();
            if (data.success) {
                setStatus('↺ Analytics reset. Video kept.', 'text-slate-400');
                // Clear local chart data
                if (occupancyChart) { occupancyChart.data.labels = []; occupancyChart.data.datasets[0].data = []; occupancyChart.update(); }
                if (dwellChart) { dwellChart.data.labels = []; dwellChart.data.datasets[0].data = []; dwellChart.update(); }
                document.getElementById('dwell-body').innerHTML = '';
                document.getElementById('traffic-body').innerHTML = '';
                document.getElementById('traffic-highlights').innerHTML = '';
                document.getElementById('matrix-body').innerHTML = '';
                document.getElementById('common-path').textContent = 'Collecting data…';
                document.getElementById('common-path').classList.add('italic');
                ['stat-current','stat-peak','stat-avg','stat-unique'].forEach(id => document.getElementById(id).textContent = '0');
                document.getElementById('stat-dwell').textContent = '0.0s';
            }
        } catch (e) {
            setStatus('Cannot reach backend.', 'text-red-400');
        }
    });

    // ── Toggle heatmap ────────────────────────────────────────────────────────
    const heatmapContainer = document.getElementById('heatmap-container');
    document.getElementById('btn-toggle-heatmap').addEventListener('click', function() {
        heatmapVisible = !heatmapVisible;
        heatmapContainer.classList.toggle('hidden', !heatmapVisible);
        this.textContent = heatmapVisible ? 'Hide Heatmap' : 'Show Heatmap';
    });

    // ── Reset heatmap only ────────────────────────────────────────────────────
    document.getElementById('btn-reset-heatmap').addEventListener('click', async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/control/reset_heatmap`, { method: 'POST' });
            const data = await res.json();
            if (data.success) setStatus('Heatmap cleared.', 'text-slate-400');
        } catch (e) {
            setStatus('Cannot reach backend.', 'text-red-400');
        }
    });

    // ── CSV Export ────────────────────────────────────────────────────────────
    document.getElementById('btn-export-tracking').addEventListener('click', () => {
        window.location.href = `${API_BASE_URL}/export/tracking_csv`;
    });
    document.getElementById('btn-export-occupancy').addEventListener('click', () => {
        window.location.href = `${API_BASE_URL}/export/occupancy_csv`;
    });
    document.getElementById('btn-export-zone').addEventListener('click', () => {
        window.location.href = `${API_BASE_URL}/export/zone_analysis_csv`;
    });
    document.getElementById('btn-export-matrix').addEventListener('click', () => {
        window.location.href = `${API_BASE_URL}/export/movement_matrix_csv`;
    });
});
