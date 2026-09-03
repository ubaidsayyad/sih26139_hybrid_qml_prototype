/**
 * HybridQML Platform Assistant (Rule-Based Chatbot)
 * Standalone offline AI assistant for answering architecture, benchmark,
 * dataset, and platform questions without external APIs or internet dependencies.
 */
(function() {
  'use strict';

  const INTENTS = [
    {
      id: 'platform_overview',
      patterns: [
        /what\s+is\s+(this\s+)?platform/i,
        /what\s+does\s+this\s+do/i,
        /tell\s+me\s+about\s+(this\s+|the\s+)?(project|platform|app|prototype|system)/i,
        /what\s+is\s+hybridqml/i,
        /^overview$/i,
        /purpose\s+of\s+(this\s+|the\s+)?(site|project|platform)/i
      ],
      keywords: ['platform', 'hybridqml', 'what does', 'purpose', 'overview', 'prototype', 'project', 'disease detection'],
      response: `<strong>HybridQML</strong> is an end-to-end medical research prototype for early disease detection, benchmarked on cellular morphology data (breast cancer diagnostics). It bridges classical clinical preprocessing (StandardScaler + PCA) with an experimental 4-qubit <strong>Variational Quantum Classifier (VQC)</strong>, enabling clinicians and researchers to train the hybrid model, benchmark it directly against classical baselines, evaluate model-agnostic explainability (SHAP and permutation feature importance), and run real-time predictions.`
    },
    {
      id: 'what_is_qml',
      patterns: [
        /what\s+is\s+(quantum\s+machine\s+learning|qml)/i,
        /how\s+does\s+(quantum\s+computing|qml)\s+work/i,
        /explain\s+qml/i,
        /what\s+is\s+a\s+vqc/i,
        /variational\s+quantum/i
      ],
      keywords: ['quantum machine learning', 'qml', 'quantum ml', 'vqc', 'quantum computing', 'ansatz', 'qubit', 'qubits'],
      response: `<strong>Quantum Machine Learning (QML)</strong> maps data into high-dimensional quantum Hilbert spaces to identify subtle non-linear patterns. In our platform, 4 PCA features are angle-encoded as <code>RY(π·x)</code> rotations across 4 qubits. Parameterized rotation gates (<code>RY</code>, <code>RZ</code>) and CNOT entangling gates transform the quantum state, and measuring the Pauli-Z observable produces the prediction probability.`
    },
    {
      id: 'quantum_vs_classical_accuracy',
      patterns: [
        /why\s+is\s+the\s+quantum\s+model\s+less\s+accurate/i,
        /quantum\s+(vs|versus)\s+classical/i,
        /accuracy\s+comparison/i,
        /is\s+quantum\s+(better|more\s+accurate)/i,
        /accuracy\s+lower/i,
        /less\s+accurate/i,
        /why\s+is\s+classical\s+better/i
      ],
      keywords: ['less accurate', 'accuracy', 'quantum vs classical', 'benchmark', 'comparison', 'random forest', 'logistic regression', 'performance'],
      response: `Full classical models (Random Forest, Logistic Regression) train on all <strong>30 raw morphological features</strong> and achieve ~95.8%–96.5% accuracy. Our NISQ quantum circuit operates on <strong>4 qubits (4 PCA features)</strong> and achieves <strong>90.2% accuracy</strong>.<br><br>Crucially, on our <em>fair matched baseline</em> (classical model trained on the exact same 4 PCA features at 95.1%), the gap is much narrower. We transparently report real NISQ hardware constraints without marketing hype.`
    },
    {
      id: 'patient_data_privacy',
      patterns: [
        /is\s+this\s+using\s+real\s+patient\s+data/i,
        /real\s+patient\s+(data|records|information)/i,
        /data\s+privacy/i,
        /is\s+data\s+(saved|stored|persisted)/i,
        /hipaa/i
      ],
      keywords: ['real patient', 'privacy', 'patient data', 'records', 'hipaa', 'confidential', 'disk write', 'storage', 'anonymized'],
      response: `The platform trains and benchmarks on the widely recognized, open-source <strong>Breast Cancer Wisconsin (Diagnostic)</strong> dataset from UC Irvine (569 anonymized cellular samples).<br><br>No private patient medical records are stored. All batch CSV uploads are evaluated strictly in volatile RAM (<code>io.StringIO</code>) with <strong>zero disk writes</strong> and immediate in-memory eviction.`
    },
    {
      id: 'medical_device_disclaimer',
      patterns: [
        /is\s+this\s+a\s+certified\s+medical\s+device/i,
        /certified\s+medical/i,
        /fda/i,
        /cdsco/i,
        /ce\s+mark/i,
        /medical\s+device/i,
        /clinical\s+use/i,
        /can\s+doctors\s+use\s+this\s+for\s+diagnosis/i
      ],
      keywords: ['certified', 'medical device', 'fda', 'cdsco', 'clinical', 'diagnosis', 'approved', 'diagnostic tool'],
      response: `<strong>No.</strong> This platform is an academic research and demonstration prototype built for <strong>Smart India Hackathon 2026 (Problem Statement SIH26139)</strong>. It is <strong>not certified by the FDA, CDSCO, or CE</strong> as a medical diagnostic device. It is intended for exploratory decision-support research. Always consult licensed medical professionals for clinical diagnosis.`
    },
    {
      id: 'try_prediction',
      patterns: [
        /how\s+do\s+i\s+try\s+a\s+prediction/i,
        /how\s+to\s+predict/i,
        /make\s+a\s+prediction/i,
        /test\s+a\s+patient/i,
        /single\s+prediction/i,
        /try\s+it/i
      ],
      keywords: ['try prediction', 'try it', 'how to predict', 'run prediction', 'test patient', 'sliders', 'single prediction'],
      response: `You can test predictions right now:<br>1. Head to the <strong><a href="/predict" style="color:var(--primary-cyan);text-decoration:underline;">Try it</a></strong> page from the top navigation.<br>2. Adjust clinical sliders (Mean Radius, Texture, Concavity, etc.) or pick a one-click sample preset (<em>"Typical Benign"</em> or <em>"High-Risk Malignant"</em>).<br>3. Click <strong>"Run Live Dual-Model Prediction"</strong> to view simultaneous Quantum VQC and Classical predictions with confidence scores and clinical risk levels.`
    },
    {
      id: 'parameter_shift_rule',
      patterns: [
        /what\s+is\s+the\s+parameter[\s-]shift\s+rule/i,
        /parameter[\s-]shift/i,
        /quantum\s+gradient/i,
        /how\s+is\s+it\s+trained/i,
        /gradient\s+computation/i
      ],
      keywords: ['parameter shift', 'parameter-shift', 'gradient', 'backprop', 'quantum training', 'analytic gradient'],
      response: `The <strong>parameter-shift rule</strong> is an exact analytic formula for computing quantum circuit gradients on real quantum hardware without numerical approximation errors or classical backpropagation:<br><code>∂⟨Z⟩/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2</code><br>By shifting each gate angle forward and backward by π/2 and measuring expectation values, the classical optimizer (Adam) updates variational angles directly from quantum state evaluations.`
    },
    {
      id: 'team_info',
      patterns: [
        /who\s+built\s+this/i,
        /who\s+is\s+the\s+team/i,
        /who\s+made\s+this/i,
        /team\s+platinum/i,
        /about\s+the\s+team/i,
        /creators/i,
        /authors/i,
        /contributors/i
      ],
      keywords: ['who built', 'who made', 'team', 'team platinum', 'authors', 'creators', 'members', 'college', 'kits', 'ramtek'],
      response: `Engineered by <strong>TEAM PLATINUM</strong> from Kavikulguru Institute of Technology and Science (KITS), Ramtek for SIH 2026:<br>• <strong>Pranjal Madkam</strong> (Team Lead & Architecture)<br>• <strong>Ubaid Sayyad</strong> (Backend & DB)<br>• <strong>Saloni Bopche</strong> (Frontend UI)<br>• <strong>Sheshandri Lajurkar</strong> (Machine Learning)<br>• <strong>Roshani Tijare</strong> (Explainability XAI)<br>• <strong>Anshu Ambule</strong> (Testing & Docs)<br><br>Learn more on our dedicated <strong><a href="/about" style="color:var(--primary-cyan);text-decoration:underline;">About Us</a></strong> page!`
    },
    {
      id: 'upload_data',
      patterns: [
        /how\s+do\s+i\s+upload\s+patient\s+data/i,
        /upload\s+(batch|data|csv|file)/i,
        /batch\s+processing/i,
        /csv\s+format/i
      ],
      keywords: ['upload', 'batch', 'csv', 'patient data upload', 'dataset upload', 'file upload'],
      response: `Go to <strong><a href="/upload" style="color:var(--primary-cyan);text-decoration:underline;">Upload Batch</a></strong> in the navigation bar:<br>1. Download our pre-formatted template with <strong>"Download Sample Batch CSV"</strong>.<br>2. Drag and drop your <code>.csv</code> file into the dropzone.<br>3. Click <strong>"Analyze Batch"</strong> to get instant patient-level concordance, risk triage, and dual quantum-classical diagnostic agreements processed completely in RAM.`
    },
    {
      id: 'quantum_backends',
      patterns: [
        /what\s+quantum\s+backends\s+(do\s+you|are)\s+support(ed)?/i,
        /backends/i,
        /ibm\s+quantum/i,
        /qiskit/i,
        /simulator/i,
        /hardware/i
      ],
      keywords: ['backends', 'quantum backends', 'ibm', 'simulator', 'qiskit', 'hardware', 'statevector'],
      response: `We support three execution backends:<br>1. <strong>Local NumPy Simulator (<code>simulator</code>)</strong>: Ultra-low latency (~0.6ms/pass) offline engine for fast hackathon demos.<br>2. <strong>Qiskit Simulator (<code>qiskit_simulator</code>)</strong>: Gate-level simulation and statevector validation.<br>3. <strong>IBM Quantum Hardware (<code>ibm_hardware</code>)</strong>: Direct execution on superconducting cloud QPUs via IBM Quantum Runtime.`
    },
    {
      id: 'greetings',
      patterns: [
        /^(hi|hello|hey|greetings|howdy|sup)\b/i,
        /who\s+are\s+you/i,
        /^help\b/i
      ],
      keywords: ['hello', 'hi', 'hey', 'help', 'who are you', 'good morning', 'good afternoon'],
      response: `Hello! 👋 I am the <strong>HybridQML Platform Assistant</strong>. I can answer questions regarding our hybrid quantum-classical architecture, explainability (XAI), benchmarking against classical baselines, or how to try predictions. What would you like to know?`
    },
    {
      id: 'thanks',
      patterns: [
        /^(thanks|thank\s+you|awesome|great|cool|good\s+job)/i
      ],
      keywords: ['thanks', 'thank you', 'great', 'awesome', 'cool', 'appreciate'],
      response: `You're very welcome! Feel free to ask anything else about the platform or test a prediction on the <strong><a href="/predict" style="color:var(--primary-cyan);text-decoration:underline;">Try it</a></strong> page.`
    }
  ];

  const FALLBACK_RESPONSE = "I'm not sure about that one — try asking about the architecture, the quantum vs classical comparison, data privacy, or how to use the dashboard. You can also check the API Docs page or ask the team directly.";

  function matchIntent(rawText) {
    if (!rawText || !rawText.trim()) return null;
    const text = rawText.trim().toLowerCase();

    // 1. Exact / Regex pattern matching
    for (const intent of INTENTS) {
      if (intent.patterns) {
        for (const pattern of intent.patterns) {
          if (pattern.test(text)) {
            return intent.response;
          }
        }
      }
    }

    // 2. Keyword score matching
    let bestScore = 0;
    let bestResponse = null;

    for (const intent of INTENTS) {
      if (intent.keywords) {
        let score = 0;
        for (const kw of intent.keywords) {
          if (text.includes(kw.toLowerCase())) {
            score += kw.length; // weight longer specific phrases higher
          }
        }
        if (score > bestScore) {
          bestScore = score;
          bestResponse = intent.response;
        }
      }
    }

    if (bestScore >= 3 && bestResponse) {
      return bestResponse;
    }

    return FALLBACK_RESPONSE;
  }

  // Expose matcher for unit testing
  window.HybridQMLAssistant = {
    matchIntent: matchIntent,
    INTENTS: INTENTS,
    FALLBACK_RESPONSE: FALLBACK_RESPONSE
  };

  // DOM Elements & Initialization
  document.addEventListener('DOMContentLoaded', () => {
    const widget = document.getElementById('platform-assistant-widget');
    const toggleBtn = document.getElementById('assistant-toggle-btn');
    const closeBtn = document.getElementById('assistant-close-btn');
    const panel = document.getElementById('assistant-panel');
    const messagesContainer = document.getElementById('assistant-messages');
    const form = document.getElementById('assistant-form');
    const input = document.getElementById('assistant-input');

    if (!widget || !toggleBtn || !panel || !messagesContainer || !input) return;

    let isOpen = false;

    function openPanel() {
      isOpen = true;
      widget.classList.add('is-open');
      panel.setAttribute('aria-hidden', 'false');
      toggleBtn.setAttribute('aria-expanded', 'true');
      setTimeout(() => input.focus(), 260);
      scrollToBottom();
    }

    function closePanel() {
      isOpen = false;
      widget.classList.remove('is-open');
      panel.setAttribute('aria-hidden', 'true');
      toggleBtn.setAttribute('aria-expanded', 'false');
    }

    toggleBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (isOpen) {
        closePanel();
      } else {
        openPanel();
      }
    });

    if (closeBtn) {
      closeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        closePanel();
      });
    }

    // Close when clicking outside panel
    document.addEventListener('click', (e) => {
      if (isOpen && !widget.contains(e.target)) {
        closePanel();
      }
    });

    // Prevent closing when clicking inside panel
    panel.addEventListener('click', (e) => {
      e.stopPropagation();
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && isOpen) {
        closePanel();
      }
    });

    function scrollToBottom() {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function appendUserMessage(text) {
      const msgDiv = document.createElement('div');
      msgDiv.className = 'assistant-msg assistant-msg-user';
      msgDiv.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
      messagesContainer.appendChild(msgDiv);
      scrollToBottom();
    }

    function showTypingIndicator() {
      const indicatorDiv = document.createElement('div');
      indicatorDiv.className = 'assistant-msg assistant-msg-bot assistant-typing-indicator';
      indicatorDiv.innerHTML = `
        <div class="typing-bubble">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      `;
      messagesContainer.appendChild(indicatorDiv);
      scrollToBottom();
      return indicatorDiv;
    }

    function appendBotMessage(htmlContent) {
      const msgDiv = document.createElement('div');
      msgDiv.className = 'assistant-msg assistant-msg-bot';
      msgDiv.innerHTML = `<div class="msg-bubble"><p>${htmlContent}</p></div>`;
      messagesContainer.appendChild(msgDiv);
      scrollToBottom();
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
    }

    function handleUserInput(query) {
      const text = (query || input.value || '').trim();
      if (!text) return;

      appendUserMessage(text);
      if (input) input.value = '';

      // Typing indicator with artificial delay (450-650ms)
      const indicator = showTypingIndicator();
      const delay = Math.floor(Math.random() * 200) + 450;

      setTimeout(() => {
        if (indicator && indicator.parentNode) {
          indicator.parentNode.removeChild(indicator);
        }
        const answer = matchIntent(text);
        appendBotMessage(answer);
      }, delay);
    }

    if (form) {
      form.addEventListener('submit', (e) => {
        e.preventDefault();
        handleUserInput();
      });
    }

    // Suggested Question Chips Click Handler (delegated)
    messagesContainer.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip-btn');
      if (chip) {
        const query = chip.getAttribute('data-query') || chip.innerText;
        handleUserInput(query);
      }
    });
  });
})();
