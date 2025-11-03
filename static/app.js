// State
let currentSessionId = null;
let eventSource = null;
let chatMode = 'question';
const API_BASE = 'http://localhost:8000';

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    setInterval(pollForUpdates, 1000);

    // Auto-resize textarea
    const chatInput = document.getElementById('chatInput');
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    });

    // Send on Enter (without Shift)
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});

// ============================================================================
// Modal Management
// ============================================================================

function openNewAnalysisModal() {
    document.getElementById('newAnalysisModal').classList.add('active');
}

function closeNewAnalysisModal() {
    document.getElementById('newAnalysisModal').classList.remove('active');
}

function closeArtifactsModal() {
    document.getElementById('artifactsModal').classList.remove('active');
}

// ============================================================================
// Chat Mode
// ============================================================================

function setChatMode(mode) {
    chatMode = mode;

    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    const placeholder = mode === 'question'
        ? 'Ask a question about the analysis...'
        : 'Describe how to refine the analysis...';
    document.getElementById('chatInput').placeholder = placeholder;
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
    const sessionList = document.getElementById('sessionList');
    sessionList.innerHTML = '';

    sessions.forEach(session => {
        const item = document.createElement('div');
        item.className = 'session-item';
        if (session.session_id === currentSessionId) {
            item.classList.add('active');
        }

        item.innerHTML = `
            <div class="session-item-id">${session.session_id.substring(0, 8)}...</div>
            <div class="session-item-task">${session.task}</div>
        `;

        item.onclick = () => loadSession(session.session_id);
        sessionList.appendChild(item);
    });
}

async function loadSession(sessionId) {
    currentSessionId = sessionId;

    // Update active state
    document.querySelectorAll('.session-item').forEach(item => {
        item.classList.toggle('active', item.textContent.includes(sessionId.substring(0, 8)));
    });

    // Clear chat and show system message
    clearChat();
    addSystemMessage(`Loading session ${sessionId.substring(0, 8)}...`);

    // Enable chat
    enableChat();

    // Show artifacts button
    document.getElementById('artifactsBtn').style.display = 'block';

    // Update title
    document.getElementById('chatTitle').textContent = `Session ${sessionId.substring(0, 8)}`;
}

// ============================================================================
// New Analysis
// ============================================================================

async function startNewAnalysis(event) {
    event.preventDefault();

    const fileInput = document.getElementById('dataFile');
    const taskInput = document.getElementById('taskInput');
    const maxIterations = document.getElementById('maxIterations').value;

    const file = fileInput.files[0];
    const task = taskInput.value.trim();

    if (!file || !task) {
        alert('Please provide both a file and a task');
        return;
    }

    closeNewAnalysisModal();
    clearChat();

    try {
        // Show starting message
        addSystemMessage('Starting new analysis...');

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

        // Update title
        document.getElementById('chatTitle').textContent = `Session ${currentSessionId.substring(0, 8)}`;

        // Show user message
        addUserMessage(task);

        // Start streaming
        startReasoningStream(currentSessionId);

        // Enable chat
        enableChat();

        // Show artifacts button
        document.getElementById('artifactsBtn').style.display = 'block';

        // Reload sessions
        loadSessions();

        // Reset form
        fileInput.value = '';
        taskInput.value = '';

    } catch (error) {
        console.error('Error starting analysis:', error);
        addSystemMessage('Error starting analysis: ' + error.message);
    }
}

// ============================================================================
// Reasoning Stream
// ============================================================================

function startReasoningStream(sessionId) {
    if (eventSource) {
        eventSource.close();
    }

    addSystemMessage('Agent is working... <span class="spinner"></span>');

    eventSource = new EventSource(`${API_BASE}/analysis/${sessionId}/stream`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'completion') {
            handleCompletion(data);
        } else {
            addReasoningCard(data);
        }
    };

    eventSource.onerror = (error) => {
        console.error('EventSource error:', error);
        eventSource.close();
    };
}

let lastReasoningCount = 0;

async function pollForUpdates() {
    if (!currentSessionId) return;

    try {
        const response = await fetch(`${API_BASE}/analysis/${currentSessionId}/status`);
        const status = await response.json();

        if (status.reasoning_entries > lastReasoningCount) {
            lastReasoningCount = status.reasoning_entries;
        }

        if (status.status === 'completed' && eventSource) {
            handleCompletion({ status: status.status, is_valid: status.is_valid });
        }
    } catch (error) {
        // Silent fail for polling
    }
}

function handleCompletion(data) {
    console.log('Analysis completed:', data);

    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    addSystemMessage(`✅ Analysis ${data.status}! ${data.is_valid ? 'Results validated.' : ''}`);
    addSystemMessage('📄 Click "View Artifacts" to see reports, code, and plots.');

    loadSessions();
}

// ============================================================================
// Chat Messages
// ============================================================================

function clearChat() {
    document.getElementById('chatMessages').innerHTML = '';
}

function enableChat() {
    document.getElementById('chatInput').disabled = false;
    document.getElementById('sendBtn').disabled = false;
}

function addMessage(type, sender, text) {
    const chatMessages = document.getElementById('chatMessages');

    // Remove empty state
    const emptyState = chatMessages.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    const avatarMap = {
        user: '👤',
        agent: '🤖',
        system: 'ℹ️'
    };

    const message = document.createElement('div');
    message.className = 'message';
    message.innerHTML = `
        <div class="message-header">
            <div class="message-avatar ${type}">${avatarMap[type]}</div>
            <div class="message-meta">
                <div class="message-sender">${sender}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        </div>
        <div class="message-content">
            <div class="message-text">${text}</div>
        </div>
    `;

    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addUserMessage(text) {
    addMessage('user', 'You', text);
}

function addAgentMessage(text) {
    addMessage('agent', 'Agent', text);
}

function addSystemMessage(text) {
    addMessage('system', 'System', text);
}

function addReasoningCard(data) {
    const chatMessages = document.getElementById('chatMessages');

    // Remove empty state
    const emptyState = chatMessages.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    const message = document.createElement('div');
    message.className = 'message';

    const cardId = `reasoning-${Date.now()}-${Math.random()}`;

    message.innerHTML = `
        <div class="message-header">
            <div class="message-avatar agent">🤖</div>
            <div class="message-meta">
                <div class="message-sender">Agent (Iteration ${data.iteration})</div>
                <div class="message-time">${new Date(data.timestamp).toLocaleTimeString()}</div>
            </div>
        </div>
        <div class="reasoning-card" id="${cardId}">
            <div class="reasoning-card-header" onclick="toggleReasoningCard('${cardId}')">
                <div class="reasoning-card-title">💭 ${data.node}</div>
                <div class="reasoning-card-icon">▼</div>
            </div>
            <div class="reasoning-card-body">${escapeHtml(data.thought)}</div>
        </div>
    `;

    chatMessages.appendChild(message);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function toggleReasoningCard(cardId) {
    const card = document.getElementById(cardId);
    card.classList.toggle('expanded');
}

// ============================================================================
// Send Message
// ============================================================================

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message || !currentSessionId) return;

    addUserMessage(message);
    input.value = '';
    input.style.height = 'auto';

    document.getElementById('sendBtn').disabled = true;

    try {
        if (chatMode === 'question') {
            // Ask question
            const formData = new FormData();
            formData.append('question', message);

            const response = await fetch(`${API_BASE}/analysis/${currentSessionId}/question`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            addAgentMessage(result.answer);

        } else {
            // Refine analysis
            addSystemMessage('Starting refinement...');

            const response = await fetch(`${API_BASE}/analysis/${currentSessionId}/refine`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    refinement_prompt: message,
                    max_iterations: 2
                })
            });

            if (eventSource) {
                eventSource.close();
            }
            startReasoningStream(currentSessionId);
        }

    } catch (error) {
        console.error('Error:', error);
        addSystemMessage('Error: ' + error.message);
    } finally {
        document.getElementById('sendBtn').disabled = false;
    }
}

// ============================================================================
// Artifacts
// ============================================================================

async function showArtifacts() {
    if (!currentSessionId) return;

    const modal = document.getElementById('artifactsModal');
    const content = document.getElementById('artifactsContent');

    content.innerHTML = '<p>Loading artifacts...</p>';
    modal.classList.add('active');

    try {
        const response = await fetch(`${API_BASE}/analysis/${currentSessionId}/artifacts`);
        const artifacts = await response.json();

        let html = '';

        // Reports
        if (artifacts.reports && artifacts.reports.length > 0) {
            html += '<h3 style="margin-bottom: 10px;">📄 Reports</h3>';
            artifacts.reports.forEach(file => {
                html += `
                    <div style="margin-bottom: 10px;">
                        <a href="${API_BASE}/analysis/${currentSessionId}/files/${file.filename}" download style="color: #3498db;">
                            ${file.filename}
                        </a>
                    </div>
                `;
            });
            html += '<hr style="margin: 20px 0;">';
        }

        // Code
        if (artifacts.code && artifacts.code.length > 0) {
            html += '<h3 style="margin-bottom: 10px;">💻 Code</h3>';
            artifacts.code.forEach(file => {
                html += `
                    <div style="margin-bottom: 10px;">
                        <a href="${API_BASE}/analysis/${currentSessionId}/files/${file.filename}" download style="color: #3498db;">
                            ${file.filename}
                        </a>
                    </div>
                `;
            });
            html += '<hr style="margin: 20px 0;">';
        }

        // Plots
        if (artifacts.plots && artifacts.plots.length > 0) {
            html += '<h3 style="margin-bottom: 10px;">📊 Plots</h3>';
            html += '<div class="plots-grid">';
            artifacts.plots.forEach(file => {
                html += `
                    <div class="plot-thumb">
                        <img src="${API_BASE}/analysis/${currentSessionId}/files/${file.filename}" alt="${file.filename}">
                        <div style="padding: 8px; font-size: 11px; text-align: center;">
                            <a href="${API_BASE}/analysis/${currentSessionId}/files/${file.filename}" download style="color: #3498db;">
                                📥 ${file.filename}
                            </a>
                        </div>
                    </div>
                `;
            });
            html += '</div>';
        }

        if (!html) {
            html = '<p>No artifacts available yet.</p>';
        }

        content.innerHTML = html;

    } catch (error) {
        console.error('Error loading artifacts:', error);
        content.innerHTML = '<p>Error loading artifacts.</p>';
    }
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
