// State management
let currentSessionId = null;
let eventSource = null;
let chatMode = 'question'; // 'question' or 'refine'
const API_BASE = 'http://localhost:8000';

// DOM elements
const startAnalysisBtn = document.getElementById('startAnalysisBtn');
const newChatBtn = document.getElementById('newChatBtn');
const dataFileInput = document.getElementById('dataFile');
const taskInput = document.getElementById('taskInput');
const maxIterationsInput = document.getElementById('maxIterations');
const reasoningContent = document.getElementById('reasoningContent');
const chatInput = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');
const chatHistory = document.getElementById('chatHistory');
const sessionList = document.getElementById('sessionList');

// ============================================================================
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeChatMode();
    loadSessions();

    // Poll for reasoning updates
    setInterval(pollForUpdates, 1000);
});

// ============================================================================
// Tab Management
// ============================================================================

function initializeTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.dataset.tab;

            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.getElementById(`tab-${tabName}`).classList.add('active');
        });
    });
}

// ============================================================================
// Chat Mode Management
// ============================================================================

function initializeChatMode() {
    document.querySelectorAll('.chat-mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            chatMode = btn.dataset.mode;

            document.querySelectorAll('.chat-mode-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update placeholder
            if (chatMode === 'question') {
                chatInput.placeholder = 'Ask a question about the analysis...';
            } else {
                chatInput.placeholder = 'Describe how to refine the analysis...';
            }
        });
    });
}

// ============================================================================
// Session Management
// ============================================================================

async function loadSessions() {
    try {
        const response = await fetch(`${API_BASE}/sessions`);
        const data = await response.json();

        if (data.sessions && data.sessions.length > 0) {
            displaySessions(data.sessions);
        }
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

function displaySessions(sessions) {
    sessionList.innerHTML = '';

    sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = 'session-item';
        if (session.session_id === currentSessionId) {
            item.classList.add('active');
        }

        const statusClass = session.status === 'completed' ? 'status-completed' :
                           session.status === 'failed' ? 'status-failed' : 'status-processing';

        item.innerHTML = `
            <div class="session-item-header">
                <span class="session-item-id">${session.session_id.substring(0, 8)}...</span>
                <span class="session-item-status ${statusClass}">${session.status}</span>
            </div>
            <div class="session-item-task">${session.task}</div>
        `;

        item.addEventListener('click', () => loadSession(session.session_id));

        sessionList.appendChild(item);
    });
}

async function loadSession(sessionId) {
    currentSessionId = sessionId;

    // Update UI
    document.querySelectorAll('.session-item').forEach(item => {
        item.classList.remove('active');
    });
    event.currentTarget.classList.add('active');

    // Load session data
    try {
        const statusResponse = await fetch(`${API_BASE}/analysis/${sessionId}/status`);
        const status = await statusResponse.json();

        // Load artifacts
        await loadArtifacts(sessionId);

        // Enable chat if completed
        if (status.status === 'completed') {
            chatInput.disabled = false;
            chatSendBtn.disabled = false;
        }

        addChatMessage('system', `Loaded session ${sessionId.substring(0, 8)}...`);
    } catch (error) {
        console.error('Error loading session:', error);
        addChatMessage('system', 'Error loading session: ' + error.message);
    }
}

// ============================================================================
// New Analysis
// ============================================================================

startAnalysisBtn.addEventListener('click', async () => {
    const file = dataFileInput.files[0];
    const task = taskInput.value.trim();
    const maxIterations = maxIterationsInput.value;

    if (!file || !task) {
        alert('Please select a file and enter an analysis task');
        return;
    }

    startAnalysisBtn.disabled = true;
    startAnalysisBtn.textContent = 'Starting...';

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('task', task);
        formData.append('max_iterations', maxIterations);

        const response = await fetch(`${API_BASE}/analysis/new`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        currentSessionId = result.session_id;

        // Clear previous content
        clearReasoningStream();
        clearArtifacts();
        chatHistory.innerHTML = '';

        // Start streaming reasoning
        startReasoningStream(currentSessionId);

        // Enable chat
        chatInput.disabled = false;
        chatSendBtn.disabled = false;

        addChatMessage('system', 'Analysis started. Watching for reasoning updates...');

        // Reload sessions list
        loadSessions();

    } catch (error) {
        console.error('Error starting analysis:', error);
        alert('Error starting analysis: ' + error.message);
    } finally {
        startAnalysisBtn.disabled = false;
        startAnalysisBtn.textContent = 'Start Analysis';
    }
});

// ============================================================================
// New Chat
// ============================================================================

newChatBtn.addEventListener('click', () => {
    if (confirm('Start a new chat? This will clear the current session.')) {
        resetSession();
    }
});

function resetSession() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    currentSessionId = null;
    chatHistory.innerHTML = '';
    chatInput.value = '';
    chatInput.disabled = true;
    chatSendBtn.disabled = true;

    clearReasoningStream();
    clearArtifacts();

    reasoningContent.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">🤔</div>
            <p>Waiting for analysis to start...</p>
        </div>
    `;

    document.getElementById('reasoningStatus').textContent = '';

    // Deselect sessions
    document.querySelectorAll('.session-item').forEach(item => {
        item.classList.remove('active');
    });
}

// ============================================================================
// Reasoning Stream
// ============================================================================

function startReasoningStream(sessionId) {
    if (eventSource) {
        eventSource.close();
    }

    document.getElementById('reasoningStatus').innerHTML = '<span class="loading-text">Live <span class="spinner"></span></span>';

    eventSource = new EventSource(`${API_BASE}/analysis/${sessionId}/stream`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'completion') {
            handleCompletion(data);
        } else {
            addReasoningEntry(data);
        }
    };

    eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        document.getElementById('reasoningStatus').textContent = '(Disconnected)';
        eventSource.close();
    };
}

function addReasoningEntry(data) {
    const emptyState = reasoningContent.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    const entry = document.createElement('div');
    entry.className = 'reasoning-entry';
    entry.innerHTML = `
        <div class="reasoning-entry-header">
            <span class="reasoning-node">${data.node}</span>
            <span class="reasoning-iteration">Iteration ${data.iteration}</span>
        </div>
        <div class="reasoning-thought">${escapeHtml(data.thought)}</div>
        <div class="reasoning-timestamp">${formatTimestamp(data.timestamp)}</div>
    `;

    reasoningContent.appendChild(entry);
    reasoningContent.scrollTop = reasoningContent.scrollHeight;
}

function handleCompletion(data) {
    console.log('Analysis completed:', data);

    document.getElementById('reasoningStatus').innerHTML = '<span style="color: #27ae60;">✓ Complete</span>';

    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    // Load artifacts
    loadArtifacts(currentSessionId);

    addChatMessage('system', `Analysis ${data.status}! Artifacts are now available.`);

    // Reload sessions list
    loadSessions();
}

// ============================================================================
// Polling for Updates (for real-time reasoning)
// ============================================================================

let lastReasoningCount = 0;

async function pollForUpdates() {
    if (!currentSessionId) return;

    try {
        const response = await fetch(`${API_BASE}/analysis/${currentSessionId}/status`);
        const status = await response.json();

        const currentCount = status.reasoning_entries || 0;

        // If new reasoning entries, fetch them
        if (currentCount > lastReasoningCount) {
            await fetchNewReasoningEntries();
            lastReasoningCount = currentCount;
        }

        // Check if completed
        if (status.status === 'completed' && document.getElementById('reasoningStatus').textContent.includes('Live')) {
            handleCompletion({ status: status.status, is_valid: status.is_valid });
        }

    } catch (error) {
        // Silently ignore polling errors
    }
}

async function fetchNewReasoningEntries() {
    // This is already handled by SSE, but kept for fallback
}

// ============================================================================
// Artifacts
// ============================================================================

async function loadArtifacts(sessionId) {
    try {
        const response = await fetch(`${API_BASE}/analysis/${sessionId}/artifacts`);
        const artifacts = await response.json();

        // Load latest of each type
        await loadLatestArtifact(sessionId, 'report', artifacts.reports);
        await loadLatestArtifact(sessionId, 'reasoning', artifacts.reasoning);
        await loadLatestArtifact(sessionId, 'code', artifacts.code);
        await loadPlots(sessionId, artifacts.plots);

    } catch (error) {
        console.error('Error loading artifacts:', error);
    }
}

async function loadLatestArtifact(sessionId, type, fileList) {
    const tabContent = document.getElementById(`tab-${type}`);

    if (!fileList || fileList.length === 0) {
        tabContent.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">${type === 'report' ? '📄' : type === 'reasoning' ? '🧠' : '💻'}</div>
                <p>No ${type} available yet</p>
            </div>
        `;
        return;
    }

    // Show all versions
    let html = '<div class="artifact-version-list">';
    html += `<h3 style="margin-bottom: 15px;">Available Versions (${fileList.length})</h3>`;

    for (let i = 0; i < fileList.length; i++) {
        const file = fileList[i];
        const isLatest = i === 0;

        html += `
            <div class="artifact-version-item ${isLatest ? 'active' : ''}"
                 onclick="loadArtifactVersion('${sessionId}', '${type}', '${file.filename}')">
                <strong>${file.filename}</strong> ${isLatest ? '(Latest)' : ''}
                <br>
                <small>Modified: ${new Date(file.modified).toLocaleString()}</small>
                <br>
                <a href="${API_BASE}/analysis/${sessionId}/files/${file.filename}" download class="download-btn" style="display: inline-block; margin-top: 5px;">
                    📥 Download
                </a>
            </div>
        `;
    }

    html += '</div>';
    html += `<div id="${type}-content"></div>`;

    tabContent.innerHTML = html;

    // Load latest version content
    await loadArtifactVersion(sessionId, type, fileList[0].filename);
}

async function loadArtifactVersion(sessionId, type, filename) {
    try {
        const response = await fetch(`${API_BASE}/analysis/${sessionId}/files/${filename}`);
        const content = await response.text();

        const contentDiv = document.getElementById(`${type}-content`);
        contentDiv.innerHTML = `<div class="artifact-content">${escapeHtml(content)}</div>`;

        // Update active state
        document.querySelectorAll(`#tab-${type} .artifact-version-item`).forEach(item => {
            item.classList.remove('active');
        });
        event?.currentTarget?.classList.add('active');

    } catch (error) {
        console.error(`Error loading ${type}:`, error);
    }
}

async function loadPlots(sessionId, plotList) {
    const tabContent = document.getElementById('tab-plots');

    if (!plotList || plotList.length === 0) {
        tabContent.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <p>No plots generated yet</p>
            </div>
        `;
        return;
    }

    let html = '<div class="plots-grid">';

    plotList.forEach(plot => {
        html += `
            <div class="plot-item">
                <img src="${API_BASE}/analysis/${sessionId}/files/${plot.filename}" alt="${plot.filename}">
                <div class="plot-item-footer">
                    <span>${plot.filename}</span>
                    <a href="${API_BASE}/analysis/${sessionId}/files/${plot.filename}" download class="plot-download-btn">
                        📥 Download
                    </a>
                </div>
            </div>
        `;
    });

    html += '</div>';
    tabContent.innerHTML = html;
}

function clearArtifacts() {
    ['report', 'reasoning', 'code', 'plots'].forEach(tab => {
        const tabContent = document.getElementById(`tab-${tab}`);
        const iconMap = { report: '📄', reasoning: '🧠', code: '💻', plots: '📊' };
        tabContent.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">${iconMap[tab]}</div>
                <p>Processing...</p>
            </div>
        `;
    });
}

// ============================================================================
// Chat
// ============================================================================

chatSendBtn.addEventListener('click', async () => {
    const message = chatInput.value.trim();

    if (!message) {
        return;
    }

    if (!currentSessionId) {
        alert('No active session. Please start an analysis first.');
        return;
    }

    addChatMessage('user', message);
    chatInput.value = '';
    chatSendBtn.disabled = true;

    try {
        if (chatMode === 'question') {
            // Ask question without re-running
            const formData = new FormData();
            formData.append('question', message);

            const response = await fetch(`${API_BASE}/analysis/${currentSessionId}/question`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            addChatMessage('assistant', result.answer);

        } else {
            // Refine analysis (re-run agent)
            const response = await fetch(`${API_BASE}/analysis/${currentSessionId}/refine`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    refinement_prompt: message,
                    max_iterations: 2
                })
            });

            const result = await response.json();

            addChatMessage('system', 'Refinement started. Watch the reasoning stream for updates...');

            // Restart streaming for refinement
            if (eventSource) {
                eventSource.close();
            }
            startReasoningStream(currentSessionId);
        }

    } catch (error) {
        console.error('Error processing chat message:', error);
        addChatMessage('system', 'Error: ' + error.message);
    } finally {
        chatSendBtn.disabled = false;
    }
});

// Allow Enter to send (Shift+Enter for new line)
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatSendBtn.click();
    }
});

function addChatMessage(type, message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    messageDiv.textContent = message;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// ============================================================================
// Reasoning
// ============================================================================

function clearReasoningStream() {
    reasoningContent.innerHTML = `
        <div class="empty-state">
            <div class="empty-state-icon">🤔</div>
            <p>Starting analysis...</p>
        </div>
    `;
    lastReasoningCount = 0;
}

// ============================================================================
// Utility Functions
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}
