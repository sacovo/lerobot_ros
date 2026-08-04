// Helper to get a consistent DOM ID for camera topics
function getCameraId(topic) {
    let clean = topic.replace(/\./g, '/');
    if (clean.startsWith('/')) {
        clean = clean.substring(1);
    }
    return `cam-${clean.replace(/\//g, '-')}`;
}

// State variables
let state = {
    bagPath: '',
    configPath: '',
    repoId: '',
    resume: false,
    startTimeLimit: 0,
    endTimeLimit: 0,
    duration: 0,
    currentTime: 0,
    playing: false,
    playbackSpeed: 1.0,
    episodes: [], // List of { id, start_time, end_time, task }
    cameras: [], // List of camera topic names
    loadingFrame: false,
    pendingFrameTime: null,
    configTopics: null,
    graphPoints: [], // List of { time, actions: [], observations: [] }
    recordingEpisode: false,
    autoStartVal: null,
    editingEpisode: null
};

// Playback interval handle
let playbackTimer = null;
let lastTickTime = 0;

// Bag combobox state
let allBags = []; // [{ path, rel_path }], as scanned from /api/bags
let filteredBags = []; // Currently visible (search-filtered) subset of allBags
let bagListOpen = false;
let activeBagIndex = -1;

// DOM Elements
const configPathInput = document.getElementById('config-path');
const repoIdInput = document.getElementById('repo-id');
const resumeDsCheckbox = document.getElementById('resume-ds');
const bagCombobox = document.getElementById('bag-combobox');
const bagSearchInput = document.getElementById('bag-search');
const bagComboboxList = document.getElementById('bag-combobox-list');
const customBagPathInput = document.getElementById('custom-bag-path');
const loadBagBtn = document.getElementById('load-bag-btn');
const cameraContainer = document.getElementById('camera-container');
const telemetryTable = document.getElementById('telemetry-table');
const startTimeInput = document.getElementById('start-time');
const endTimeInput = document.getElementById('end-time');
const setStartBtn = document.getElementById('set-start-btn');
const setEndBtn = document.getElementById('set-end-btn');
const taskNameInput = document.getElementById('task-name');
const addEpisodeBtn = document.getElementById('add-episode-btn');
const episodesTable = document.getElementById('episodes-table');
const convertBtn = document.getElementById('convert-btn');
const timelineRuler = document.getElementById('timeline-ruler');
const scrubLine = document.getElementById('scrub-line');
const currentTimeBadge = document.getElementById('current-time-badge');
const displayTime = document.getElementById('display-time');
const displayDuration = document.getElementById('display-duration');
const playBtn = document.getElementById('play-btn');
const playbackSpeedSelect = document.getElementById('playback-speed');
const statusModal = document.getElementById('status-modal');
const statusProgressFill = document.getElementById('status-progress-fill');
const statusMessage = document.getElementById('status-message');
const modalActions = document.getElementById('modal-actions');
const closeModalBtn = document.getElementById('close-modal-btn');
const cancelModalBtn = document.getElementById('cancel-modal-btn');
const warningBanner = document.getElementById('warning-banner');
const autoAnnotateBtn = document.getElementById('auto-annotate-btn');
const cancelAnnotateBtn = document.getElementById('cancel-annotate-btn');

// Playback buttons
const stepBackLargeBtn = document.getElementById('step-back-large-btn');
const stepBackBtn = document.getElementById('step-back-btn');
const stepForwardBtn = document.getElementById('step-forward-btn');
const stepForwardLargeBtn = document.getElementById('step-forward-large-btn');

// Initialize
window.addEventListener('DOMContentLoaded', async () => {
    // Restore config path and theme from localStorage
    const savedConfig = localStorage.getItem('configPath');
    if (savedConfig) {
        configPathInput.value = savedConfig;
    }
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
    } else if (savedTheme === 'dark') {
        document.body.classList.remove('light-theme');
    }

    // Load server-side default config
    try {
        const response = await fetch('/api/config');
        const data = await response.json();
        if (data.default_config) {
            configPathInput.placeholder = data.default_config;
            if (!savedConfig) {
                configPathInput.value = data.default_config;
                localStorage.setItem('configPath', data.default_config);
            }
        }
    } catch (err) {
        console.error('Failed to load server config:', err);
    }

    scanBags();
    setupEventListeners();
});

// Scan workspace for bags
async function scanBags() {
    try {
        const response = await fetch('/api/bags');
        const data = await response.json();
        allBags = data.bags || [];
        filteredBags = allBags;
        renderBagList(filteredBags);
    } catch (err) {
        console.error('Failed to scan workspace bags:', err);
    }
}

// Render the (optionally filtered) bag list, grouped by parent folder
// (e.g. a date-named subfolder) so bags recorded on different days/sessions
// are easy to tell apart at a glance.
function renderBagList(bags) {
    bagComboboxList.innerHTML = '';
    activeBagIndex = -1;

    if (bags.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'bag-combobox-empty';
        empty.textContent = allBags.length === 0 ? 'No bags found under scan root.' : 'No bags match your search.';
        bagComboboxList.appendChild(empty);
        return;
    }

    let lastGroup = undefined;
    bags.forEach((bag, idx) => {
        const slashIdx = bag.rel_path.lastIndexOf('/');
        const group = slashIdx === -1 ? null : bag.rel_path.slice(0, slashIdx);
        if (group !== lastGroup) {
            const label = document.createElement('div');
            label.className = 'bag-group-label';
            label.textContent = group || 'Ungrouped';
            bagComboboxList.appendChild(label);
            lastGroup = group;
        }
        const item = document.createElement('div');
        item.className = 'bag-combobox-item';
        item.textContent = group ? bag.rel_path.slice(slashIdx + 1) : bag.rel_path;
        item.title = bag.rel_path;
        item.dataset.index = String(idx);
        bagComboboxList.appendChild(item);
    });
}

function filterBags(query) {
    const q = query.trim().toLowerCase();
    filteredBags = q ? allBags.filter(b => b.rel_path.toLowerCase().includes(q)) : allBags;
    renderBagList(filteredBags);
}

function openBagList() {
    bagComboboxList.classList.add('open');
    bagListOpen = true;
}

function closeBagList() {
    bagComboboxList.classList.remove('open');
    bagListOpen = false;
    activeBagIndex = -1;
}

function selectBag(bag) {
    if (!bag) return;
    customBagPathInput.value = bag.path;
    bagSearchInput.value = bag.rel_path;
    closeBagList();
}

function updateActiveBagItem(items) {
    items.forEach((el, i) => el.classList.toggle('active', i === activeBagIndex));
    if (activeBagIndex >= 0 && items[activeBagIndex]) {
        items[activeBagIndex].scrollIntoView({ block: 'nearest' });
    }
}

function setupEventListeners() {
    // Save config path on change
    configPathInput.addEventListener('input', () => {
        localStorage.setItem('configPath', configPathInput.value);
    });

    // Bag selection combobox (searchable, grouped by folder)
    bagSearchInput.addEventListener('focus', () => {
        bagSearchInput.select();
        renderBagList(filteredBags);
        openBagList();
    });

    bagSearchInput.addEventListener('input', () => {
        filterBags(bagSearchInput.value);
        openBagList();
    });

    bagSearchInput.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (!bagListOpen) {
                openBagList();
                return;
            }
            const items = bagComboboxList.querySelectorAll('.bag-combobox-item');
            activeBagIndex = Math.min(activeBagIndex + 1, items.length - 1);
            updateActiveBagItem(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            const items = bagComboboxList.querySelectorAll('.bag-combobox-item');
            activeBagIndex = Math.max(activeBagIndex - 1, 0);
            updateActiveBagItem(items);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            const items = bagComboboxList.querySelectorAll('.bag-combobox-item');
            if (activeBagIndex >= 0 && items[activeBagIndex]) {
                const idx = parseInt(items[activeBagIndex].dataset.index, 10);
                selectBag(filteredBags[idx]);
            }
        } else if (e.key === 'Escape') {
            closeBagList();
            bagSearchInput.blur();
        }
    });

    // Use mousedown (not click) + preventDefault so the input doesn't lose
    // focus/close the list before the selection is registered.
    bagComboboxList.addEventListener('mousedown', (e) => {
        e.preventDefault();
        const item = e.target.closest('.bag-combobox-item');
        if (!item) return;
        const idx = parseInt(item.dataset.index, 10);
        selectBag(filteredBags[idx]);
    });

    document.addEventListener('click', (e) => {
        if (!bagCombobox.contains(e.target)) {
            closeBagList();
        }
    });

    // Load Bag
    loadBagBtn.addEventListener('click', loadBag);

    // Set markers to current time
    setStartBtn.addEventListener('click', () => {
        startTimeInput.value = state.currentTime.toFixed(2);
    });
    setEndBtn.addEventListener('click', () => {
        endTimeInput.value = state.currentTime.toFixed(2);
    });

    // Add Episode
    addEpisodeBtn.addEventListener('click', addEpisode);

    // Timeline Clicking / Scrubbing
    timelineRuler.addEventListener('mousedown', startScrubbing);
    
    // Playback Controls
    playBtn.addEventListener('click', togglePlayback);
    
    playbackSpeedSelect.addEventListener('change', () => {
        state.playbackSpeed = parseFloat(playbackSpeedSelect.value);
    });

    // Step buttons
    document.getElementById('step-back-large-btn').addEventListener('click', () => seekRelative(-1.0));
    document.getElementById('step-back-btn').addEventListener('click', () => seekRelative(-0.05));
    document.getElementById('step-forward-btn').addEventListener('click', () => seekRelative(0.05));
    document.getElementById('step-forward-large-btn').addEventListener('click', () => seekRelative(1.0));

    // Convert
    convertBtn.addEventListener('click', startConversion);
    closeModalBtn.addEventListener('click', () => {
        statusModal.classList.remove('active');
    });
    cancelModalBtn.addEventListener('click', async () => {
        try {
            cancelModalBtn.disabled = true;
            cancelModalBtn.textContent = 'Cancelling...';
            await fetch('/api/convert/cancel', { method: 'POST' });
        } catch (e) {
            console.error('Failed to request cancel:', e);
        }
    });

    // Auto Annotation
    autoAnnotateBtn.addEventListener('click', handleAutoAnnotate);
    cancelAnnotateBtn.addEventListener('click', cancelAutoAnnotate);

    // Global keyboard shortcut Ctrl+H
    window.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key.toLowerCase() === 'h') {
            e.preventDefault();
            handleAutoAnnotate();
        }
    });

    // Redraw graph on window resize
    window.addEventListener('resize', () => {
        drawGraph();
    });

    // Theme toggle
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const isLight = document.body.classList.toggle('light-theme');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
            drawGraph();
        });
    }
}

// Load Bag Details from API
async function loadBag() {
    const bagPath = customBagPathInput.value.trim();
    if (!bagPath) {
        alert('Please enter or select a bag path.');
        return;
    }

    loadBagBtn.disabled = true;
    loadBagBtn.textContent = 'Loading...';

    try {
        const configPath = configPathInput.value.trim();
        const response = await fetch(`/api/bag-info?path=${encodeURIComponent(bagPath)}&config=${encodeURIComponent(configPath)}`);
        const data = await response.json();
        
        if (data.error) {
            alert(`Error loading bag: ${data.error}`);
            return;
        }

        // Set state
        state.bagPath = data.path;
        state.duration = data.duration;
        state.startTimeLimit = data.start_time;
        state.endTimeLimit = data.end_time;
        state.currentTime = 0;
        state.episodes = [];
        state.playing = false;
        state.configTopics = data.config_topics || null;
        state.graphPoints = [];
        state.recordingEpisode = false;
        state.autoStartVal = null;
        state.editingEpisode = null;

        // Reset UI
        displayDuration.textContent = state.duration.toFixed(2);
        updatePlaybackUI();
        renderEpisodesTable();
        renderTimelineBlocks();
        
        // Reset Auto Annotator UI
        autoAnnotateBtn.textContent = '⏺ Start Episode (Ctrl+H)';
        autoAnnotateBtn.className = 'btn-primary w-full';
        cancelAnnotateBtn.style.display = 'none';
        
        // Reset Add/Edit Episode Button
        addEpisodeBtn.textContent = 'Save Episode Range';
        addEpisodeBtn.className = 'btn-success w-full';
        
        // Show graph container
        const graphContainer = document.getElementById('graph-container');
        if (graphContainer) {
            graphContainer.style.display = 'block';
        }
        
        // Find cameras in topics list
        state.cameras = data.topics
            .filter(t => t.type === 'sensor_msgs/msg/Image' || t.type === 'sensor_msgs/msg/CompressedImage')
            .map(t => t.name);

        // Pre-create image elements in camera container
        cameraContainer.innerHTML = '';
        if (state.cameras.length === 0) {
            cameraContainer.innerHTML = '<div class="empty-state">No cameras detected in bag.</div>';
        } else {
            state.cameras.forEach(topic => {
                const wrapper = document.createElement('div');
                wrapper.className = 'camera-viewport';
                
                const img = document.createElement('img');
                img.id = getCameraId(topic);
                img.alt = topic;
                img.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360"><rect width="100%" height="100%" fill="%23141621"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="Outfit" fill="%239aa0a6" font-size="16">Waiting for frame...</text></svg>';
                
                const label = document.createElement('div');
                label.className = 'camera-label';
                label.textContent = topic;
                
                wrapper.appendChild(img);
                wrapper.appendChild(label);
                cameraContainer.appendChild(wrapper);
            });
        }

        // Load the first frame
        loadFrame(0);

    } catch (err) {
        alert(`Failed to query bag info: ${err.message}`);
    } finally {
        loadBagBtn.disabled = false;
        loadBagBtn.textContent = 'Load Bag';
    }
}

// Request frame from API
async function loadFrame(timeSec) {
    if (state.loadingFrame) {
        // Debounce: store pending frame request
        state.pendingFrameTime = timeSec;
        return;
    }

    state.loadingFrame = true;
    const configPath = configPathInput.value.trim();
    
    try {
        const response = await fetch(`/api/frame?path=${encodeURIComponent(state.bagPath)}&time=${timeSec}&config=${encodeURIComponent(configPath)}`);
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            const errMsg = errData.detail || `HTTP ${response.status}`;
            console.error('Frame retrieval error:', errMsg);
            alert(`Error retrieving frame: ${errMsg}`);
            if (state.playing) {
                togglePlayback();
            }
            return;
        }
        const data = await response.json();

        // Update warnings display
        if (data.warnings && data.warnings.length > 0) {
            warningBanner.textContent = data.warnings.join('\n');
            warningBanner.style.display = 'block';
        } else {
            warningBanner.style.display = 'none';
        }

        // 1. Update Camera Feeds
        if (data.cameras) {
            for (const [topic, base64Data] of Object.entries(data.cameras)) {
                // Topic names map back matching slashes
                const img = document.getElementById(getCameraId(topic));
                if (img) {
                    img.src = base64Data;
                }
            }
        }

        // 2. Update Telemetry Table
        updateTelemetryTable(data.telemetry);

        // 3. Update Graph Points & Render Graph
        updateGraphData(timeSec, data.telemetry);

    } catch (err) {
        console.error('Failed to load frame:', err);
    } finally {
        state.loadingFrame = false;
        
        // Execute pending frame request if queued
        if (state.pendingFrameTime !== null) {
            const nextTime = state.pendingFrameTime;
            state.pendingFrameTime = null;
            loadFrame(nextTime);
        }
    }
}

// Update the telemetry display table
function updateTelemetryTable(telemetry) {
    if (!telemetry || Object.keys(telemetry).length === 0) {
        telemetryTable.innerHTML = `
            <thead>
                <tr>
                    <th>Topic / Field</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td colspan="2" class="empty-state">No telemetry at this timestamp.</td>
                </tr>
            </tbody>`;
        return;
    }

    let tbodyHtml = '';
    for (const [topic, data] of Object.entries(telemetry)) {
        tbodyHtml += `
            <tr class="topic-header-row">
                <td colspan="2" style="background: rgba(255,255,255,0.02); font-weight:700;">${topic.startsWith('/') ? topic : '/' + topic.replace(/\./g, '/')}</td>
            </tr>`;
        
        for (const [key, val] of Object.entries(data)) {
            let valStr = '';
            if (Array.isArray(val)) {
                valStr = `[${val.map(x => typeof x === 'number' ? x.toFixed(4) : x).join(', ')}]`;
            } else if (typeof val === 'object' && val !== null) {
                valStr = JSON.stringify(val);
            } else if (typeof val === 'number') {
                valStr = val.toFixed(4);
            } else {
                valStr = val.toString();
            }

            tbodyHtml += `
                <tr>
                    <td style="padding-left: 24px; color: var(--text-secondary);">${key}</td>
                    <td><pre>${valStr}</pre></td>
                </tr>`;
        }
    }

    telemetryTable.innerHTML = `
        <thead>
            <tr>
                <th>Topic / Field</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            ${tbodyHtml}
        </tbody>`;
}

// Seek directly to relative offset
function seekRelative(offset) {
    if (!state.bagPath) return;
    seekTo(Math.max(0, Math.min(state.duration, state.currentTime + offset)));
}

// Seek to target timestamp
function seekTo(timeSec) {
    state.currentTime = timeSec;
    
    // Update playhead displays
    displayTime.textContent = timeSec.toFixed(2);
    currentTimeBadge.textContent = `${timeSec.toFixed(2)}s`;
    
    const percentage = (timeSec / state.duration) * 100;
    scrubLine.style.left = `${percentage}%`;
    
    // Sync telemetry graph playhead line
    drawGraph();
    
    loadFrame(timeSec);
}

// Playback Logic
function togglePlayback() {
    if (!state.bagPath) return;

    if (state.playing) {
        // Pause
        state.playing = false;
        clearInterval(playbackTimer);
        playBtn.textContent = '▶ Play';
        playBtn.className = 'btn-primary';
    } else {
        // Play
        state.playing = true;
        lastTickTime = performance.now();
        playBtn.textContent = '⏸ Pause';
        playBtn.className = 'btn-secondary';
        
        playbackTimer = setInterval(playbackTick, 50); // 20 FPS updates
    }
}

function playbackTick() {
    if (!state.playing) return;

    const now = performance.now();
    const elapsedSec = (now - lastTickTime) / 1000.0;
    lastTickTime = now;

    let nextTime = state.currentTime + elapsedSec * state.playbackSpeed;
    
    if (nextTime >= state.duration) {
        nextTime = state.duration;
        togglePlayback(); // Stop at end of bag
    }

    seekTo(nextTime);
}

function updatePlaybackUI() {
    displayTime.textContent = state.currentTime.toFixed(2);
    currentTimeBadge.textContent = `${state.currentTime.toFixed(2)}s`;
    scrubLine.style.left = '0%';
}

// Timeline Scrub dragging
function startScrubbing(e) {
    if (!state.bagPath) return;
    
    if (state.playing) {
        togglePlayback(); // Pause during scrub
    }
    
    scrub(e);
    
    window.addEventListener('mousemove', scrub);
    window.addEventListener('mouseup', stopScrubbing);
}

function scrub(e) {
    const rect = timelineRuler.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const percentage = x / rect.width;
    seekTo(percentage * state.duration);
}

function stopScrubbing() {
    window.removeEventListener('mousemove', scrub);
    window.removeEventListener('mouseup', stopScrubbing);
}

// Add/Save Episode Range annotation
function addEpisode() {
    if (!state.bagPath) {
        alert('Please load a bag file first.');
        return;
    }

    const startTime = parseFloat(startTimeInput.value);
    const endTime = parseFloat(endTimeInput.value);
    const taskName = taskNameInput.value.trim();

    if (isNaN(startTime) || isNaN(endTime) || startTime < 0 || endTime <= startTime || endTime > state.duration) {
        alert('Please enter valid, positive start and end times (Start < End <= Duration).');
        return;
    }

    if (!taskName) {
        alert('Please enter a task name.');
        return;
    }

    if (state.editingEpisode !== null) {
        // Update existing episode
        state.editingEpisode.start_time = startTime;
        state.editingEpisode.end_time = endTime;
        state.editingEpisode.task = taskName;
        
        state.editingEpisode = null;
        addEpisodeBtn.textContent = 'Save Episode Range';
        addEpisodeBtn.className = 'btn-success w-full';
    } else {
        // Add new episode
        const episode = {
            id: state.episodes.length + 1,
            start_time: startTime,
            end_time: endTime,
            task: taskName
        };
        state.episodes.push(episode);
    }
    
    // Sort episodes chronologically
    state.episodes.sort((a, b) => a.start_time - b.start_time);
    
    // Reindex IDs
    state.episodes.forEach((ep, idx) => ep.id = idx + 1);

    // Refresh UI
    renderEpisodesTable();
    renderTimelineBlocks();

    // Clear inputs for next episode entry
    startTimeInput.value = '';
    endTimeInput.value = '';
    taskNameInput.value = '';
}

function deleteEpisode(id) {
    const episodeToDelete = state.episodes.find(ep => ep.id === id);
    if (state.editingEpisode && state.editingEpisode === episodeToDelete) {
        state.editingEpisode = null;
        addEpisodeBtn.textContent = 'Save Episode Range';
        addEpisodeBtn.className = 'btn-success w-full';
        startTimeInput.value = '';
        endTimeInput.value = '';
        taskNameInput.value = '';
    }

    state.episodes = state.episodes.filter(ep => ep.id !== id);
    // Reindex
    state.episodes.forEach((ep, idx) => ep.id = idx + 1);
    
    renderEpisodesTable();
    renderTimelineBlocks();
}

// Render episode ranges into the list table
function renderEpisodesTable() {
    if (state.episodes.length === 0) {
        episodesTable.innerHTML = `
            <thead>
                <tr>
                    <th>#</th>
                    <th>Range (s)</th>
                    <th>Task</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td colspan="4" class="empty-state">No episodes added yet.</td>
                </tr>
            </tbody>`;
        return;
    }

    let tbodyHtml = '';
    state.episodes.forEach(ep => {
        tbodyHtml += `
            <tr class="episode-row" onclick="selectEpisode(${ep.id})">
                <td>${ep.id}</td>
                <td>${ep.start_time.toFixed(2)} - ${ep.end_time.toFixed(2)}s</td>
                <td>${ep.task}</td>
                <td>
                    <button class="btn-danger" style="height:24px; padding:0 8px; font-size:11px;" onclick="event.stopPropagation(); deleteEpisode(${ep.id})">Delete</button>
                </td>
            </tr>`;
    });

    episodesTable.innerHTML = `
        <thead>
            <tr>
                <th>#</th>
                <th>Range (s)</th>
                <th>Task</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            ${tbodyHtml}
        </tbody>`;
}

// Select an episode to load its values into the form and jump to its start time
function selectEpisode(id) {
    const ep = state.episodes.find(e => e.id === id);
    if (!ep) return;
    
    // Jump playhead to start time
    seekTo(ep.start_time);
    
    // Load values into form inputs
    startTimeInput.value = ep.start_time.toFixed(2);
    endTimeInput.value = ep.end_time.toFixed(2);
    taskNameInput.value = ep.task;
    
    // Set editing reference
    state.editingEpisode = ep;
    
    // Update button text and style
    addEpisodeBtn.textContent = 'Update Episode';
    addEpisodeBtn.className = 'btn-primary w-full';
}

// Draw visual shaded intervals on timeline scrubber
function renderTimelineBlocks() {
    // Remove existing blocks
    const existingBlocks = timelineRuler.querySelectorAll('.timeline-block');
    existingBlocks.forEach(b => b.remove());

    state.episodes.forEach(ep => {
        const startPercent = (ep.start_time / state.duration) * 100;
        const endPercent = (ep.end_time / state.duration) * 100;
        const widthPercent = endPercent - startPercent;

        const block = document.createElement('div');
        block.className = 'timeline-block';
        block.style.left = `${startPercent}%`;
        block.style.width = `${widthPercent}%`;
        block.title = `${ep.task} (${ep.start_time.toFixed(2)} - ${ep.end_time.toFixed(2)}s)`;
        block.textContent = ep.task;

        timelineRuler.appendChild(block);
    });
}

// Dataset Conversion Trigger & Poller
async function startConversion() {
    if (!state.bagPath) {
        alert('Please load a bag file first.');
        return;
    }

    if (state.episodes.length === 0) {
        alert('Please add at least one episode before converting.');
        return;
    }

    const configPath = configPathInput.value.trim();
    const repoId = repoIdInput.value.trim();
    const resume = resumeDsCheckbox.checked;

    if (!configPath || !repoId) {
        alert('Please enter valid Configuration Path and Dataset Name values.');
        return;
    }

    // Open progress modal
    statusModal.classList.add('active');
    statusProgressFill.className = 'progress-bar-fill'; // Reset state classes
    statusProgressFill.style.width = '0%';
    statusMessage.textContent = 'Sending conversion request to backend...';
    
    // Show cancel button, hide close button during startup
    closeModalBtn.style.display = 'none';
    cancelModalBtn.style.display = 'inline-block';
    cancelModalBtn.disabled = false;
    cancelModalBtn.textContent = 'Cancel Conversion';
    modalActions.style.display = 'flex';

    try {
        const response = await fetch('/api/convert', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                bag_path: state.bagPath,
                config: configPath,
                repo_id: repoId,
                resume: resume,
                annotations: state.episodes.map(ep => ({
                    start_time: ep.start_time,
                    end_time: ep.end_time,
                    task: ep.task
                }))
            })
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            statusMessage.textContent = `Error: ${data.detail || response.statusText}`;
            modalActions.style.display = 'flex';
            closeModalBtn.style.display = 'inline-block';
            cancelModalBtn.style.display = 'none';
            return;
        }

        // Start polling status
        setTimeout(pollStatus, 500);

    } catch (err) {
        statusMessage.textContent = `Network failure: ${err.message}`;
        modalActions.style.display = 'flex';
        closeModalBtn.style.display = 'inline-block';
        cancelModalBtn.style.display = 'none';
    }
}

async function pollStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        statusMessage.textContent = data.message;
        
        if (data.total > 0) {
            const percentage = (data.progress / data.total) * 100;
            statusProgressFill.style.width = `${percentage}%`;
        }

        if (data.running) {
            closeModalBtn.style.display = 'none';
            cancelModalBtn.style.display = 'inline-block';
            modalActions.style.display = 'flex';
            // Keep polling
            setTimeout(pollStatus, 500);
        } else {
            // Completed, failed, or cancelled
            modalActions.style.display = 'flex';
            closeModalBtn.style.display = 'inline-block';
            cancelModalBtn.style.display = 'none';
            
            statusProgressFill.style.width = '100%';
            if (data.state === 'error') {
                statusProgressFill.className = 'progress-bar-fill error';
            } else if (data.state === 'done') {
                statusProgressFill.className = 'progress-bar-fill success';
            }
        }
    } catch (err) {
        statusMessage.textContent = `Status Polling Error: ${err.message}`;
        modalActions.style.display = 'flex';
        closeModalBtn.style.display = 'inline-block';
        cancelModalBtn.style.display = 'none';
    }
}

// Extract numbers recursively from telemetry objects/arrays
function extractNumbers(obj, list) {
    if (typeof obj === 'number') {
        list.push(obj);
    } else if (Array.isArray(obj)) {
        obj.forEach(item => extractNumbers(item, list));
    } else if (typeof obj === 'object' && obj !== null) {
        Object.values(obj).forEach(val => extractNumbers(val, list));
    }
}

// Update the stored graph data and redraw
function updateGraphData(timeSec, telemetry) {
    if (!telemetry || !state.configTopics) {
        // Even if there's no telemetry data yet, redraw the graph to update the vertical playhead line
        drawGraph();
        return;
    }
    
    // Check if point for this time already exists
    const exists = state.graphPoints.some(pt => Math.abs(pt.time - timeSec) < 1e-4);
    if (exists) {
        drawGraph();
        return;
    }
    
    const actions = [];
    const observations = [];
    
    for (const [topic, val] of Object.entries(telemetry)) {
        const info = state.configTopics[topic];
        if (!info) continue;
        
        if (info.tag === 'action') {
            extractNumbers(val, actions);
        } else if (info.tag === 'observation') {
            extractNumbers(val, observations);
        }
    }
    
    if (actions.length > 0 || observations.length > 0) {
        state.graphPoints.push({
            time: timeSec,
            actions: actions,
            observations: observations
        });
        state.graphPoints.sort((a, b) => a.time - b.time);
    }
    
    drawGraph();
}

function drawGraph() {
    const canvas = document.getElementById('telemetry-graph');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Ensure the backing store matches display size
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const isLightTheme = document.body.classList.contains('light-theme');
    const playheadColor = isLightTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(255, 255, 255, 0.6)';
    const textColor = isLightTheme ? 'rgba(0, 0, 0, 0.4)' : 'rgba(255, 255, 255, 0.3)';
    
    if (state.graphPoints.length === 0) {
        ctx.fillStyle = textColor;
        ctx.font = '12px Outfit, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Scrub or play to load actions (red) & observations (blue) graph', canvas.width / 2, canvas.height / 2);
        
        // Draw playhead if duration exists
        if (state.duration > 0) {
            const playheadX = (state.currentTime / state.duration) * canvas.width;
            ctx.beginPath();
            ctx.moveTo(playheadX, 0);
            ctx.lineTo(playheadX, canvas.height);
            ctx.strokeStyle = playheadColor;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
        return;
    }
    
    // Find min and max y-values
    let minY = Infinity;
    let maxY = -Infinity;
    
    state.graphPoints.forEach(pt => {
        pt.actions.forEach(v => {
            if (v < minY) minY = v;
            if (v > maxY) maxY = v;
        });
        pt.observations.forEach(v => {
            if (v < minY) minY = v;
            if (v > maxY) maxY = v;
        });
    });
    
    if (minY === Infinity || maxY === -Infinity) {
        minY = 0;
        maxY = 1;
    } else if (Math.abs(maxY - minY) < 1e-5) {
        minY -= 1;
        maxY += 1;
    }
    
    // Plot points
    state.graphPoints.forEach(pt => {
        const x = (pt.time / state.duration) * canvas.width;
        
        // Draw actions in red
        pt.actions.forEach(v => {
            const y = canvas.height - ((v - minY) / (maxY - minY)) * canvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(239, 68, 68, 0.8)'; // red
            ctx.fill();
        });
        
        // Draw observations in blue
        pt.observations.forEach(v => {
            const y = canvas.height - ((v - minY) / (maxY - minY)) * canvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
            ctx.fillStyle = 'rgba(59, 130, 246, 0.8)'; // blue
            ctx.fill();
        });
    });
    
    // Draw vertical playhead line
    if (state.duration > 0) {
        const playheadX = (state.currentTime / state.duration) * canvas.width;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, canvas.height);
        ctx.strokeStyle = playheadColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

function handleAutoAnnotate() {
    if (!state.bagPath) {
        alert('Please load a bag file first.');
        return;
    }

    if (!state.recordingEpisode) {
        // Start new episode recording
        state.recordingEpisode = true;
        state.autoStartVal = state.currentTime;
        
        startTimeInput.value = state.currentTime.toFixed(2);
        endTimeInput.value = '';
        
        autoAnnotateBtn.textContent = '⏹ Stop Episode (Ctrl+H)';
        autoAnnotateBtn.className = 'btn-danger w-full';
        cancelAnnotateBtn.style.display = 'block';
    } else {
        // End the current episode recording
        state.recordingEpisode = false;
        endTimeInput.value = state.currentTime.toFixed(2);
        
        cancelAnnotateBtn.style.display = 'none';
        autoAnnotateBtn.textContent = '⏺ Start Episode (Ctrl+H)';
        autoAnnotateBtn.className = 'btn-primary w-full';
        
        // Make sure a task name is present; if not, fill in default
        if (!taskNameInput.value.trim()) {
            taskNameInput.value = `Episode ${state.episodes.length + 1}`;
        }
        
        // Call the save logic
        addEpisode();
        state.autoStartVal = null;
    }
}

function cancelAutoAnnotate() {
    if (state.recordingEpisode) {
        state.recordingEpisode = false;
        state.autoStartVal = null;
        startTimeInput.value = '';
        endTimeInput.value = '';
        
        cancelAnnotateBtn.style.display = 'none';
        autoAnnotateBtn.textContent = '⏺ Start Episode (Ctrl+H)';
        autoAnnotateBtn.className = 'btn-primary w-full';
    }
}
